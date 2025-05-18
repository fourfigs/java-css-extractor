"""
Content generators for the Java CSS Optimizer.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple, Union, Callable, TypeVar, Generic
from functools import lru_cache, wraps
import csscompressor
import weakref
import threading
import os
import json
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, TimeoutError
import time
import psutil
from collections import OrderedDict, defaultdict
import asyncio
import aiofiles
import orjson
from dataclasses import dataclass, field
from enum import Enum, auto
import tracemalloc
import gc
import sys
from typing_extensions import TypedDict, Protocol, runtime_checkable
import signal
from datetime import datetime, timedelta
import hashlib
import uuid
import queue
import weakref
from abc import ABC, abstractmethod

from java_css_optimizer.analyzer import StylePattern, AnalysisResult
from java_css_optimizer.config import Config
from java_css_optimizer.optimizer.constants import (
    CSS_INDENT, CSS_SELECTOR_SEPARATOR, CSS_RULE_SEPARATOR,
    JAVA_INDENT, JAVA_IMPORTS, MAX_CSS_RULES_PER_FILE,
    MAX_JAVA_BEHAVIORS_PER_FILE, MAX_CACHE_SIZE, MAX_MEMORY_USAGE,
    TEMPLATE_CACHE_SIZE, OPERATION_TIMEOUT, MAX_RETRIES,
    RETRY_DELAY, MAX_WORKERS, CACHE_CLEANUP_INTERVAL
)
from java_css_optimizer.optimizer.models import CSSRule, JavaBehavior
from java_css_optimizer.optimizer.exceptions import (
    CSSRuleLimitException, JavaBehaviorLimitException,
    TemplateLoadError, MemoryLimitExceeded, CacheLimitExceeded,
    OperationTimeout, RetryLimitExceeded, ValidationError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class CacheStats(TypedDict):
    """Cache statistics."""
    hits: int
    misses: int
    size: int
    max_size: int
    evictions: int
    hit_ratio: float
    miss_ratio: float
    avg_access_time: float
    last_cleanup: datetime
    memory_usage: int

class CacheEntry(TypedDict):
    """Cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size: int
    ttl: Optional[float]
    tags: Set[str]

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ARC = "arc"
    TWO_QUEUE = "2q"
    CLOCK = "clock"

@dataclass
class CacheConfig:
    """Cache configuration."""
    max_size: int = MAX_CACHE_SIZE
    strategy: CacheStrategy = CacheStrategy.LRU
    cleanup_interval: float = CACHE_CLEANUP_INTERVAL
    ttl: Optional[float] = None
    max_memory_usage: int = MAX_MEMORY_USAGE
    enable_compression: bool = False
    enable_encryption: bool = False
    enable_persistence: bool = False
    persistence_path: Optional[Path] = None
    tags: Set[str] = field(default_factory=set)

@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations."""
    def get(self, key: Any) -> Optional[Any]: ...
    def put(self, key: Any, value: Any, tags: Optional[Set[str]] = None) -> None: ...
    def clear(self) -> None: ...
    def get_stats(self) -> CacheStats: ...
    def invalidate_by_tag(self, tag: str) -> None: ...
    def get_by_tag(self, tag: str) -> List[Any]: ...
    def cleanup(self) -> None: ...

class AdvancedCache(Generic[K, V]):
    """Advanced cache implementation with multiple strategies and statistics."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_ratio": 0.0,
            "miss_ratio": 0.0,
            "avg_access_time": 0.0,
            "last_cleanup": datetime.now(),
            "memory_usage": 0
        }
        self._cleanup_timer = None
        self._access_times: List[float] = []
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._start_cleanup_timer()
        
        if self.config.enable_persistence and self.config.persistence_path:
            self._load_persisted_cache()
    
    def _start_cleanup_timer(self) -> None:
        """Start periodic cache cleanup."""
        def cleanup():
            try:
                with self._lock:
                    self._cleanup_expired()
                    self._update_stats()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
            finally:
                if self._cleanup_timer:
                    self._cleanup_timer = threading.Timer(
                        self.config.cleanup_interval,
                        cleanup
                    )
                    self._cleanup_timer.start()
    
    def _cleanup_expired(self) -> None:
        """Clean up expired entries."""
        if not self.config.ttl:
            return
        
        now = time.time()
        expired = [
            key for key, entry in self.cache.items()
            if now - entry["created_at"] > entry["ttl"]
        ]
        
        for key in expired:
            try:
                self._remove_entry(key)
            except Exception as e:
                logger.error(f"Error evicting cache entry {key}: {str(e)}")
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache."""
        entry = self.cache.pop(key)
        with self._stats_lock:
            self._stats["evictions"] += 1
            self._stats["memory_usage"] -= entry["size"]
        
        # Remove from tag index
        for tag in entry["tags"]:
            self._tag_index[tag].discard(key)
            if not self._tag_index[tag]:
                del self._tag_index[tag]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with strategy-specific updates."""
        start_time = time.time()
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                entry["last_accessed"] = time.time()
                entry["access_count"] += 1
                
                if self.config.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)
                elif self.config.strategy == CacheStrategy.CLOCK:
                    entry["referenced"] = True
                
                with self._stats_lock:
                    self._stats["hits"] += 1
                    self._access_times.append(time.time() - start_time)
                
                return entry["value"]
            
            with self._stats_lock:
                self._stats["misses"] += 1
                self._access_times.append(time.time() - start_time)
            return None
    
    def put(self, key: str, value: Any, tags: Optional[Set[str]] = None) -> None:
        """Put item in cache with strategy-specific eviction."""
        with self._lock:
            try:
                # Calculate entry size
                entry_size = sys.getsizeof(value)
                
                # Check memory limit
                if self._stats["memory_usage"] + entry_size > self.config.max_memory_usage:
                    self._evict_entries(entry_size)
                
                # Remove existing entry if present
                if key in self.cache:
                    self._remove_entry(key)
                
                # Create new entry
                entry = {
                    "value": value,
                    "created_at": time.time(),
                    "last_accessed": time.time(),
                    "access_count": 0,
                    "size": entry_size,
                    "ttl": self.config.ttl,
                    "tags": tags or set(),
                    "referenced": True
                }
                
                # Add to cache
                self.cache[key] = entry
                
                # Update tag index
                for tag in entry["tags"]:
                    self._tag_index[tag].add(key)
                
                # Update memory usage
                with self._stats_lock:
                    self._stats["memory_usage"] += entry_size
                
                # Persist if enabled
                if self.config.enable_persistence:
                    self._persist_entry(key, entry)
                
            except Exception as e:
                logger.error(f"Error putting cache entry {key}: {str(e)}")
                raise
    
    def _evict_entries(self, required_space: int) -> None:
        """Evict entries to make space."""
        if self.config.strategy == CacheStrategy.LRU:
            while self._stats["memory_usage"] + required_space > self.config.max_memory_usage:
                if not self.cache:
                    raise CacheLimitExceeded("Cache is full and cannot be evicted")
                self._remove_entry(next(iter(self.cache)))
        elif self.config.strategy == CacheStrategy.LFU:
            while self._stats["memory_usage"] + required_space > self.config.max_memory_usage:
                if not self.cache:
                    raise CacheLimitExceeded("Cache is full and cannot be evicted")
                key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]["access_count"]
                )
                self._remove_entry(key)
        elif self.config.strategy == CacheStrategy.CLOCK:
            while self._stats["memory_usage"] + required_space > self.config.max_memory_usage:
                if not self.cache:
                    raise CacheLimitExceeded("Cache is full and cannot be evicted")
                for key, entry in self.cache.items():
                    if not entry["referenced"]:
                        self._remove_entry(key)
                        break
                    entry["referenced"] = False
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        with self._stats_lock:
            total = self._stats["hits"] + self._stats["misses"]
            if total > 0:
                self._stats["hit_ratio"] = self._stats["hits"] / total
                self._stats["miss_ratio"] = self._stats["misses"] / total
            
            if self._access_times:
                self._stats["avg_access_time"] = sum(self._access_times) / len(self._access_times)
                self._access_times.clear()
            
            self._stats["last_cleanup"] = datetime.now()
    
    def clear(self) -> None:
        """Clear cache and reset statistics."""
        with self._lock:
            try:
                self.cache.clear()
                self._tag_index.clear()
                with self._stats_lock:
                    self._stats = {
                        "hits": 0,
                        "misses": 0,
                        "evictions": 0,
                        "hit_ratio": 0.0,
                        "miss_ratio": 0.0,
                        "avg_access_time": 0.0,
                        "last_cleanup": datetime.now(),
                        "memory_usage": 0
                    }
                if self.config.enable_persistence:
                    self._clear_persistence()
            except Exception as e:
                logger.error(f"Error clearing cache: {str(e)}")
                raise
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._stats_lock:
            return dict(self._stats)
    
    def invalidate_by_tag(self, tag: str) -> None:
        """Invalidate all entries with the given tag."""
        with self._lock:
            if tag in self._tag_index:
                for key in list(self._tag_index[tag]):
                    self._remove_entry(key)
    
    def get_by_tag(self, tag: str) -> List[Any]:
        """Get all entries with the given tag."""
        with self._lock:
            if tag in self._tag_index:
                return [
                    self.cache[key]["value"]
                    for key in self._tag_index[tag]
                    if key in self.cache
                ]
            return []
    
    def _persist_entry(self, key: str, entry: CacheEntry) -> None:
        """Persist a cache entry to disk."""
        if not self.config.persistence_path:
            return
        
        try:
            persist_path = self.config.persistence_path / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            with open(persist_path, 'w') as f:
                json.dump(entry, f)
        except Exception as e:
            logger.error(f"Error persisting cache entry {key}: {str(e)}")
    
    def _load_persisted_cache(self) -> None:
        """Load persisted cache entries from disk."""
        if not self.config.persistence_path:
            return
        
        try:
            for persist_file in self.config.persistence_path.glob('*.json'):
                try:
                    with open(persist_file) as f:
                        entry = json.load(f)
                        key = persist_file.stem
                        self.cache[key] = entry
                        for tag in entry["tags"]:
                            self._tag_index[tag].add(key)
                except Exception as e:
                    logger.error(f"Error loading persisted cache entry {persist_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading persisted cache: {str(e)}")
    
    def _clear_persistence(self) -> None:
        """Clear persisted cache entries."""
        if not self.config.persistence_path:
            return
        
        try:
            for persist_file in self.config.persistence_path.glob('*.json'):
                try:
                    persist_file.unlink()
                except Exception as e:
                    logger.error(f"Error removing persisted cache entry {persist_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error clearing persisted cache: {str(e)}")
    
    def __del__(self):
        """Clean up resources."""
        try:
            if self._cleanup_timer:
                self._cleanup_timer.cancel()
            self.clear()
        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")

def retry(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY, backoff_factor: float = 2.0,
          max_delay: float = 60.0, exceptions: Tuple[Type[Exception], ...] = (Exception,),
          on_retry: Optional[Callable[[Exception, int], None]] = None):
    """Enhanced retry decorator with exponential backoff and custom exception handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        # Calculate next delay with exponential backoff
                        current_delay = min(current_delay * backoff_factor, max_delay)
                        
                        # Call on_retry callback if provided
                        if on_retry:
                            try:
                                on_retry(e, attempt + 1)
                            except Exception as callback_error:
                                logger.error(f"Error in retry callback: {str(callback_error)}")
                        
                        # Log retry attempt
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {current_delay:.2f}s: {str(e)}"
                        )
                        
                        await asyncio.sleep(current_delay)
                        continue
                    
                    raise RetryLimitExceeded(
                        f"Operation failed after {max_retries} retries: {str(e)}"
                    )
            
            raise last_exception
        return wrapper
    return decorator

def monitor_memory(threshold: float = 0.8, cleanup_threshold: float = 0.9,
                  max_usage: int = MAX_MEMORY_USAGE, gc_threshold: int = 3):
    """Enhanced memory monitoring decorator with automatic cleanup."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            start_memory = process.memory_info().rss
            gc_count = 0
            
            try:
                # Check initial memory usage
                if start_memory > max_usage * threshold:
                    logger.warning(
                        f"High memory usage before {func.__name__}: "
                        f"{start_memory / 1024 / 1024:.2f} MB"
                    )
                    gc.collect()
                    gc_count += 1
                
                result = func(*args, **kwargs)
                
                # Check memory usage after function execution
                end_memory = process.memory_info().rss
                memory_diff = end_memory - start_memory
                
                # Log memory usage
                if memory_diff > 0:
                    logger.debug(
                        f"{func.__name__} memory usage: {memory_diff / 1024 / 1024:.2f} MB"
                    )
                
                # Check if memory usage exceeds cleanup threshold
                if end_memory > max_usage * cleanup_threshold:
                    logger.warning(
                        f"Memory usage exceeds cleanup threshold: "
                        f"{end_memory / 1024 / 1024:.2f} MB"
                    )
                    
                    # Try to free memory
                    while gc_count < gc_threshold and process.memory_info().rss > max_usage * threshold:
                        gc.collect()
                        gc_count += 1
                    
                    # If still above max usage, raise exception
                    if process.memory_info().rss > max_usage:
                        raise MemoryLimitExceeded(
                            f"Memory usage exceeds limit: {process.memory_info().rss / 1024 / 1024:.2f} MB"
                        )
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
            finally:
                # Final memory cleanup
                if gc_count < gc_threshold:
                    gc.collect()
        
        return wrapper
    return decorator

def validate_input(validator: Callable[[Any], bool], error_message: str):
    """Input validation decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValidationError(error_message)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def measure_performance(threshold: float = 1.0):
    """Performance measurement decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                execution_time = end_time - start_time
                memory_usage = end_memory - start_memory
                
                # Log performance metrics
                logger.debug(
                    f"{func.__name__} performance: "
                    f"time={execution_time:.3f}s, "
                    f"memory={memory_usage / 1024 / 1024:.2f} MB"
                )
                
                # Warn if performance is below threshold
                if execution_time > threshold:
                    logger.warning(
                        f"{func.__name__} execution time ({execution_time:.3f}s) "
                        f"exceeds threshold ({threshold}s)"
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
                
        return wrapper
    return decorator

def handle_timeout(timeout: float = OPERATION_TIMEOUT):
    """Timeout handling decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise OperationTimeout(
                    f"Operation {func.__name__} timed out after {timeout}s"
                )
        return wrapper
    return decorator

def ensure_cleanup(cleanup_func: Callable):
    """Resource cleanup decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                try:
                    cleanup_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error during cleanup: {str(e)}")
        return wrapper
    return decorator

class TemplateManager:
    """Manages template loading and caching with advanced features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.template_dir = Path(config.get('template_dir', os.path.join(os.path.dirname(__file__), 'templates')))
        self._cache = AdvancedCache(CacheConfig(
            max_size=TEMPLATE_CACHE_SIZE,
            strategy=CacheStrategy.LRU,
            ttl=3600,  # 1 hour
            enable_persistence=True,
            persistence_path=self.template_dir / 'cache'
        ))
        self._lock = threading.RLock()
        self._template_locks = {}
        self.logger = logging.getLogger(__name__)
        self._template_loaders = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Initialize metrics
        self._metrics = {
            'loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'last_error': None,
            'last_error_time': None
        }
        self._metrics_lock = threading.RLock()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Initialize template validation
        self._template_validators = {
            'components': self._validate_component_template,
            'behaviors': self._validate_behavior_template
        }
        
        # Ensure template directory exists
        self._loop.run_until_complete(self._ensure_template_dir())
    
    def _get_template_lock(self, template_type: str, template_name: str) -> threading.RLock:
        """Get or create a lock for a specific template."""
        key = f"{template_type}:{template_name}"
        with self._lock:
            if key not in self._template_locks:
                self._template_locks[key] = threading.RLock()
            return self._template_locks[key]
    
    @monitor_memory(threshold=0.7, cleanup_threshold=0.8)
    def _check_memory_usage(self) -> None:
        """Check if memory usage exceeds limit."""
        process = psutil.Process()
        if process.memory_info().rss > MAX_MEMORY_USAGE:
            # Try to free memory
            gc.collect()
            if process.memory_info().rss > MAX_MEMORY_USAGE:
                raise MemoryLimitExceeded(
                    f"Memory usage exceeds limit: {process.memory_info().rss / 1024 / 1024:.2f} MB"
                )
    
    @retry(max_retries=3, backoff_factor=2.0, max_delay=30.0)
    @handle_timeout(timeout=OPERATION_TIMEOUT)
    async def _ensure_template_dir(self) -> None:
        """Ensure template directory exists with proper error handling."""
        try:
            if not self.template_dir.exists():
                self.logger.info(f"Creating template directory: {self.template_dir}")
                self.template_dir.mkdir(parents=True, exist_ok=True)
            
            # Create default template directories
            for template_type in ['components', 'behaviors']:
                template_type_dir = self.template_dir / template_type
                if not template_type_dir.exists():
                    self.logger.info(f"Creating template type directory: {template_type_dir}")
                    template_type_dir.mkdir(exist_ok=True)
                    
                    # Create default template file
                    default_template = await self._get_default_template(template_type, 'default')
                    await self._save_template(template_type, 'default', default_template)
            
            # Create cache directory
            cache_dir = self.template_dir / 'cache'
            if not cache_dir.exists():
                cache_dir.mkdir(exist_ok=True)
        
        except Exception as e:
            self.logger.error(f"Error ensuring template directory: {str(e)}")
            raise TemplateLoadError(f"Failed to ensure template directory: {str(e)}")
    
    @retry(max_retries=3, backoff_factor=2.0, max_delay=30.0)
    @handle_timeout(timeout=OPERATION_TIMEOUT)
    async def _save_template(self, template_type: str, template_name: str, template: Dict[str, str]) -> None:
        """Save template to file with proper error handling."""
        template_path = None
        try:
            template_path = self.template_dir / template_type / f"{template_name}.json"
            
            # Validate template before saving
            if not self._validate_template(template, template_type):
                raise ValidationError(f"Invalid template structure for {template_type}/{template_name}")
            
            # Create backup of existing template
            if template_path.exists():
                backup_path = template_path.with_suffix('.json.bak')
                template_path.rename(backup_path)
            
            # Save template
            async with aiofiles.open(template_path, 'w', encoding='utf-8') as f:
                await f.write(orjson.dumps(template, option=orjson.OPT_INDENT_2).decode())
            
            # Update cache
            cache_key = f"{template_type}:{template_name}"
            self._cache.put(cache_key, template, tags={template_type, template_name})
            
        except Exception as e:
            self.logger.error(f"Error saving template {template_path}: {str(e)}")
            
            # Restore backup if exists
            if template_path and template_path.with_suffix('.json.bak').exists():
                template_path.with_suffix('.json.bak').rename(template_path)
            
            raise TemplateLoadError(f"Failed to save template: {str(e)}")
    
    @measure_performance(threshold=0.5)
    def get_template(self, template_type: str, template_name: str) -> Dict[str, str]:
        """Get template with proper resource management."""
        cache_key = f"{template_type}:{template_name}"
        template_lock = self._get_template_lock(template_type, template_name)
        
        try:
            # Check memory usage
            self._check_memory_usage()
            
            # Try to get from cache first
            with template_lock:
                template = self._cache.get(cache_key)
                if template is not None:
                    with self._metrics_lock:
                        self._metrics['cache_hits'] += 1
                    return template
                
                with self._metrics_lock:
                    self._metrics['cache_misses'] += 1
                
                # Load template
                template = self._loop.run_until_complete(
                    self._load_template(template_type, template_name)
                )
                if template:
                    self._cache.put(cache_key, template, tags={template_type, template_name})
                
                with self._metrics_lock:
                    self._metrics['loads'] += 1
                
                return template
            
        except Exception as e:
            with self._metrics_lock:
                self._metrics['errors'] += 1
                self._metrics['last_error'] = str(e)
                self._metrics['last_error_time'] = datetime.now()
            
            self.logger.error(f"Error getting template {cache_key}: {str(e)}")
            # Return default template as fallback
            return self._loop.run_until_complete(
                self._get_default_template(template_type, template_name)
            )
    
    @retry(max_retries=3, backoff_factor=2.0, max_delay=30.0)
    @handle_timeout(timeout=OPERATION_TIMEOUT)
    async def _load_template(self, template_type: str, template_name: str) -> Dict[str, str]:
        """Load template from file with proper error handling."""
        template_path = None
        try:
            template_path = self.template_dir / template_type / f"{template_name}.json"
            
            # Check if template exists
            if not template_path.exists():
                self.logger.warning(f"Template not found: {template_path}")
                return await self._get_default_template(template_type, template_name)
            
            # Try to load template
            try:
                async with aiofiles.open(template_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    template = orjson.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in template {template_path}: {str(e)}")
                return await self._get_default_template(template_type, template_name)
            except UnicodeDecodeError as e:
                self.logger.error(f"Invalid encoding in template {template_path}: {str(e)}")
                return await self._get_default_template(template_type, template_name)
            
            # Validate template
            if not self._validate_template(template, template_type):
                self.logger.error(f"Invalid template: {template_path}")
                return await self._get_default_template(template_type, template_name)
            
            return template
            
        except Exception as e:
            self.logger.error(f"Error loading template {template_path}: {str(e)}")
            return await self._get_default_template(template_type, template_name)
    
    def _validate_template(self, template: Dict[str, str], template_type: str) -> bool:
        """Validate template structure with comprehensive error handling."""
        try:
            # Get appropriate validator
            validator = self._template_validators.get(template_type)
            if not validator:
                self.logger.error(f"No validator found for template type: {template_type}")
                return False
            
            # Validate template
            return validator(template)
            
        except Exception as e:
            self.logger.error(f"Error validating template: {str(e)}")
            return False
    
    def _validate_component_template(self, template: Dict[str, str]) -> bool:
        """Validate component template structure."""
        try:
            # Check if template is a dictionary
            if not isinstance(template, dict):
                self.logger.error("Template must be a dictionary")
                return False
            
            # Check if template is empty
            if not template:
                self.logger.error("Template cannot be empty")
                return False
            
            # Check required keys
            required_keys = {'base', 'item', 'hover', 'active'}
            if not all(key in template for key in required_keys):
                self.logger.error(f"Missing required keys: {required_keys - set(template.keys())}")
                return False
            
            # Check each key-value pair
            for key, value in template.items():
                # Check key
                if not isinstance(key, str):
                    self.logger.error(f"Invalid key type: {type(key)}")
                    return False
                if not key.strip():
                    self.logger.error("Empty key found")
                    return False
                
                # Check value
                if not isinstance(value, str):
                    self.logger.error(f"Invalid value type for key {key}: {type(value)}")
                    return False
                if not value.strip():
                    self.logger.error(f"Empty value found for key {key}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating component template: {str(e)}")
            return False
    
    def _validate_behavior_template(self, template: Dict[str, str]) -> bool:
        """Validate behavior template structure."""
        try:
            # Check if template is a dictionary
            if not isinstance(template, dict):
                self.logger.error("Template must be a dictionary")
                return False
            
            # Check if template is empty
            if not template:
                self.logger.error("Template cannot be empty")
                return False
            
            # Check required keys
            required_keys = {'base', 'effect', 'setup'}
            if not all(key in template for key in required_keys):
                self.logger.error(f"Missing required keys: {required_keys - set(template.keys())}")
                return False
            
            # Check each key-value pair
            for key, value in template.items():
                # Check key
                if not isinstance(key, str):
                    self.logger.error(f"Invalid key type: {type(key)}")
                    return False
                if not key.strip():
                    self.logger.error("Empty key found")
                    return False
                
                # Check value
                if not isinstance(value, str):
                    self.logger.error(f"Invalid value type for key {key}: {type(value)}")
                    return False
                if not value.strip():
                    self.logger.error(f"Empty value found for key {key}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating behavior template: {str(e)}")
            return False
    
    async def _get_default_template(self, template_type: str, template_name: str) -> Dict[str, str]:
        """Get default template with proper error handling."""
        try:
            if template_type == 'components':
                return {
                    'base': """
                        display: block;
                        margin: 0;
                        padding: 0;
                    """,
                    'item': """
                        display: block;
                        margin: 5px 0;
                    """,
                    'hover': """
                        background-color: #f0f0f0;
                    """,
                    'active': """
                        background-color: #e0e0e0;
                    """
                }
            else:  # behaviors
                return {
                    'base': f"""
                        public void setup{template_name.title().replace('_', '')}(JComponent component) {{
                            // TODO: Implement {template_name} behavior
                        }}
                    """,
                    'effect': """
                        // TODO: Implement effect
                    """,
                    'setup': """
                        // TODO: Implement setup
                    """
                }
        except Exception as e:
            self.logger.error(f"Error getting default template: {str(e)}")
            return {
                'base': "/* Error loading template */"
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get template manager metrics."""
        with self._metrics_lock:
            return dict(self._metrics)
    
    def clear_metrics(self) -> None:
        """Clear template manager metrics."""
        with self._metrics_lock:
            self._metrics = {
                'loads': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'errors': 0,
                'last_error': None,
                'last_error_time': None
            }
    
    def __del__(self):
        """Clean up resources with proper error handling."""
        try:
            self._cache.clear()
            self._template_loaders.shutdown(wait=True)
            self._loop.close()
            tracemalloc.stop()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

class CSSGenerator:
    """Generates CSS content from analysis results with advanced features."""
    
    def __init__(self, config: Config):
        self.config = config
        self._cache = AdvancedCache(CacheConfig(
            max_size=MAX_CACHE_SIZE,
            strategy=CacheStrategy.LRU,
            ttl=3600,  # 1 hour
            enable_persistence=True,
            persistence_path=Path(config.get('cache_dir', 'cache')) / 'css'
        ))
        self._lock = threading.RLock()
        self._class_mapping_cache = AdvancedCache(CacheConfig(
            max_size=MAX_CACHE_SIZE,
            strategy=CacheStrategy.LRU,
            ttl=3600,  # 1 hour
            enable_persistence=True,
            persistence_path=Path(config.get('cache_dir', 'cache')) / 'mappings'
        ))
        self._class_mapping_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self._css_generators = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._pattern_locks = {}
        
        # Initialize metrics
        self._metrics = {
            'generated_rules': 0,
            'duplicate_rules': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'last_error': None,
            'last_error_time': None,
            'performance': {
                'avg_generation_time': 0.0,
                'max_generation_time': 0.0,
                'min_generation_time': float('inf')
            }
        }
        self._metrics_lock = threading.RLock()
        
        # Initialize template manager
        self.template_manager = TemplateManager(config)
        
        # Initialize CSS validators
        self._css_validators = {
            'property': self._validate_css_property,
            'value': self._validate_css_value,
            'selector': self._validate_css_selector
        }
    
    def _get_pattern_lock(self, pattern: StylePattern) -> threading.RLock:
        """Get or create a lock for a specific pattern."""
        key = f"{pattern.class_name}:{pattern.css_property}"
        with self._lock:
            if key not in self._pattern_locks:
                self._pattern_locks[key] = threading.RLock()
            return self._pattern_locks[key]
    
    @monitor_memory(threshold=0.7, cleanup_threshold=0.8)
    @measure_performance(threshold=1.0)
    def generate_css_content(self, analysis: AnalysisResult) -> str:
        """Generate CSS content from analysis results."""
        start_time = time.time()
        try:
            # Check memory usage
            self._check_memory_usage()
            
            # Generate cache key
            cache_key = f"css_{hash(str(analysis))}"
            
            # Check cache first
            cached_content = self._cache.get(cache_key)
            if cached_content is not None:
                with self._metrics_lock:
                    self._metrics['cache_hits'] += 1
                return cached_content
            
            with self._metrics_lock:
                self._metrics['cache_misses'] += 1
            
            # Generate CSS rules in parallel
            css_rules = []
            futures = []
            
            # Add component-specific CSS
            for pattern in analysis.style_patterns:
                if pattern.component_type and pattern.component_type in self._load_templates('components'):
                    futures.append(self._css_generators.submit(self._generate_component_css, pattern))
            
            # Add behavior-specific CSS
            for pattern in analysis.style_patterns:
                if pattern.behavior_type and pattern.behavior_type in self._load_templates('behaviors'):
                    futures.append(self._css_generators.submit(self._generate_behavior_css, pattern))
            
            # Add style-specific CSS
            for pattern in analysis.style_patterns:
                if pattern.css_property and pattern.css_value:
                    futures.append(self._css_generators.submit(self._generate_style_css, pattern))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=OPERATION_TIMEOUT)
                    if isinstance(result, list):
                        css_rules.extend(result)
                    else:
                        css_rules.append(result)
                except TimeoutError:
                    self.logger.error("CSS generation timed out")
                    continue
                except Exception as e:
                    self.logger.error(f"Error generating CSS rule: {str(e)}")
                    continue
            
            # Update metrics
            with self._metrics_lock:
                self._metrics['generated_rules'] += len(css_rules)
            
            # Check rule limit
            if len(css_rules) > MAX_CSS_RULES_PER_FILE:
                raise CSSRuleLimitException(
                    f"Too many CSS rules generated: {len(css_rules)} > {MAX_CSS_RULES_PER_FILE}"
                )
            
            # Remove duplicate rules
            original_count = len(css_rules)
            css_rules = self._remove_duplicate_rules(css_rules)
            
            # Update metrics
            with self._metrics_lock:
                self._metrics['duplicate_rules'] += original_count - len(css_rules)
            
            # Compress CSS
            css_content = CSS_RULE_SEPARATOR.join(css_rules)
            if self.config.compress_css:
                try:
                    css_content = csscompressor.compress(css_content)
                except Exception as e:
                    self.logger.error(f"Error compressing CSS: {str(e)}")
                    # Fall back to uncompressed CSS
                    self.logger.info("Falling back to uncompressed CSS")
            
            # Cache the result
            self._cache.put(cache_key, css_content, tags={'css', 'generated'})
            
            # Update performance metrics
            generation_time = time.time() - start_time
            with self._metrics_lock:
                self._metrics['performance']['avg_generation_time'] = (
                    (self._metrics['performance']['avg_generation_time'] * 
                     (self._metrics['generated_rules'] - 1) + generation_time) /
                    self._metrics['generated_rules']
                )
                self._metrics['performance']['max_generation_time'] = max(
                    self._metrics['performance']['max_generation_time'],
                    generation_time
                )
                self._metrics['performance']['min_generation_time'] = min(
                    self._metrics['performance']['min_generation_time'],
                    generation_time
                )
            
            return css_content
            
        except Exception as e:
            with self._metrics_lock:
                self._metrics['errors'] += 1
                self._metrics['last_error'] = str(e)
                self._metrics['last_error_time'] = datetime.now()
            
            self.logger.error(f"Error generating CSS content: {str(e)}")
            raise
    
    @monitor_memory(threshold=0.7, cleanup_threshold=0.8)
    def _check_memory_usage(self) -> None:
        """Check if memory usage exceeds limit."""
        process = psutil.Process()
        if process.memory_info().rss > MAX_MEMORY_USAGE:
            # Try to free memory
            gc.collect()
            if process.memory_info().rss > MAX_MEMORY_USAGE:
                raise MemoryLimitExceeded(
                    f"Memory usage exceeds limit: {process.memory_info().rss / 1024 / 1024:.2f} MB"
                )
    
    def _load_templates(self, template_type: str) -> Dict[str, Dict[str, str]]:
        """Load templates using template manager."""
        templates = {}
        try:
            template_dir = self.template_manager.template_dir / template_type
            if not template_dir.exists():
                self.logger.warning(f"Template directory not found: {template_dir}")
                return self._get_default_templates(template_type)
            
            # Load templates in parallel
            futures = []
            for template_file in template_dir.glob('*.json'):
                template_name = template_file.stem
                futures.append(
                    self._css_generators.submit(
                        self.template_manager.get_template,
                        template_type,
                        template_name
                    )
                )
            
            # Collect results
            for future in as_completed(futures):
                try:
                    template = future.result(timeout=OPERATION_TIMEOUT)
                    if template:
                        templates[template_name] = template
                except TimeoutError:
                    self.logger.error("Template loading timed out")
                    continue
                except Exception as e:
                    self.logger.error(f"Error loading template: {str(e)}")
                    continue
            
            return templates
            
        except Exception as e:
            self.logger.error(f"Error loading templates: {str(e)}")
            return self._get_default_templates(template_type)
    
    def _get_default_templates(self, template_type: str) -> Dict[str, Dict[str, str]]:
        """Get default templates with proper error handling."""
        try:
            if template_type == 'components':
                return {
                    'default': {
                        'base': """
                            display: block;
                            margin: 0;
                            padding: 0;
                        """,
                        'item': """
                            display: block;
                            margin: 5px 0;
                        """,
                        'hover': """
                            background-color: #f0f0f0;
                        """,
                        'active': """
                            background-color: #e0e0e0;
                        """
                    }
                }
            else:  # behaviors
                return {
                    'default': {
                        'base': """
                            display: block;
                        """,
                        'effect': """
                            transition: all 0.3s ease;
                        """,
                        'setup': """
                            cursor: pointer;
                        """
                    }
                }
        except Exception as e:
            self.logger.error(f"Error getting default templates: {str(e)}")
            return {
                'default': {
                    'base': "/* Error loading templates */"
                }
            }
    
    def _create_class_mapping(self, analysis: AnalysisResult) -> Dict[str, str]:
        """Create mapping between Java classes and CSS classes."""
        # Check cache first with minimal lock scope
        cache_key = f"mapping_{hash(str(analysis))}"
        mapping = self._class_mapping_cache.get(cache_key)
        if mapping is not None:
            return mapping
        
        # Create mapping outside lock
        mapping = {}
        for pattern in analysis.style_patterns:
            if pattern.component_type:
                css_class = f"{pattern.component_type}-{pattern.class_name.lower()}"
                mapping[pattern.class_name] = css_class
        
        # Update cache with lock
        self._class_mapping_cache.put(cache_key, mapping, tags={'mapping', 'class'})
        
        return mapping
    
    def _generate_component_css(self, pattern: StylePattern) -> List[str]:
        """Generate component-specific CSS rules."""
        pattern_lock = self._get_pattern_lock(pattern)
        with pattern_lock:
            try:
                css = []
                component = self._load_templates('components')[pattern.component_type]
                
                # Validate component template
                if not self._validate_component_template(component):
                    self.logger.error(f"Invalid component template: {pattern.component_type}")
                    return []
                
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
                
                # Add hover and active states
                if pattern.has_hover:
                    css.append(f".{pattern.class_name}:hover {{")
                    css.append(component.get('hover', ''))
                    css.append("}")
                
                if pattern.has_active:
                    css.append(f".{pattern.class_name}:active {{")
                    css.append(component.get('active', ''))
                    css.append("}")
                
                return css
            except Exception as e:
                self.logger.error(f"Error generating component CSS: {str(e)}")
                return []
    
    def _generate_behavior_css(self, pattern: StylePattern) -> List[str]:
        """Generate behavior-specific CSS rules."""
        pattern_lock = self._get_pattern_lock(pattern)
        with pattern_lock:
            try:
                css = []
                behavior = self._load_templates('behaviors')[pattern.behavior_type]
                
                # Validate behavior template
                if not self._validate_behavior_template(behavior):
                    self.logger.error(f"Invalid behavior template: {pattern.behavior_type}")
                    return []
                
                css.append(f".{pattern.class_name} {{")
                css.append(behavior['base'])
                css.append("}")
                
                css.append(f".{pattern.class_name}:hover {{")
                css.append(behavior['effect'])
                css.append("}")
                
                return css
            except Exception as e:
                self.logger.error(f"Error generating behavior CSS: {str(e)}")
                return []
    
    def _generate_style_css(self, pattern: StylePattern) -> str:
        """Generate style-specific CSS rule."""
        pattern_lock = self._get_pattern_lock(pattern)
        with pattern_lock:
            try:
                # Validate CSS property and value
                if not self._validate_css_property(pattern.css_property):
                    self.logger.error(f"Invalid CSS property: {pattern.css_property}")
                    return ""
                
                if not self._validate_css_value(pattern.css_value):
                    self.logger.error(f"Invalid CSS value: {pattern.css_value}")
                    return ""
                
                return f".{pattern.class_name} {{ {pattern.css_property}: {pattern.css_value}; }}"
            except Exception as e:
                self.logger.error(f"Error generating style CSS: {str(e)}")
                return ""
    
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
    
    def _validate_css_property(self, property: str) -> bool:
        """Validate CSS property."""
        try:
            # Check if property is a string
            if not isinstance(property, str):
                return False
            
            # Check if property is empty
            if not property.strip():
                return False
            
            # Check if property is valid
            if not re.match(r'^[a-zA-Z-]+$', property):
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating CSS property: {str(e)}")
            return False
    
    def _validate_css_value(self, value: str) -> bool:
        """Validate CSS value."""
        try:
            # Check if value is a string
            if not isinstance(value, str):
                return False
            
            # Check if value is empty
            if not value.strip():
                return False
            
            # Check if value is valid
            if not re.match(r'^[a-zA-Z0-9#%()\s.,-]+$', value):
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating CSS value: {str(e)}")
            return False
    
    def _validate_css_selector(self, selector: str) -> bool:
        """Validate CSS selector."""
        try:
            # Check if selector is a string
            if not isinstance(selector, str):
                return False
            
            # Check if selector is empty
            if not selector.strip():
                return False
            
            # Check if selector is valid
            if not re.match(r'^[a-zA-Z0-9_\-\.\s,>+~:]+$', selector):
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating CSS selector: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get CSS generator metrics."""
        with self._metrics_lock:
            return dict(self._metrics)
    
    def clear_metrics(self) -> None:
        """Clear CSS generator metrics."""
        with self._metrics_lock:
            self._metrics = {
                'generated_rules': 0,
                'duplicate_rules': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'errors': 0,
                'last_error': None,
                'last_error_time': None,
                'performance': {
                    'avg_generation_time': 0.0,
                    'max_generation_time': 0.0,
                    'min_generation_time': float('inf')
                }
            }
    
    def __del__(self):
        """Clean up resources with proper error handling."""
        try:
            self._cache.clear()
            self._class_mapping_cache.clear()
            self._css_generators.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

class JavaGenerator:
    """Generates Java code for behaviors that can't be handled by CSS."""
    
    def __init__(self, config: Config):
        self.config = config
        self._cache = AdvancedCache(CacheConfig(
            max_size=MAX_CACHE_SIZE,
            strategy=CacheStrategy.LRU,
            ttl=3600  # 1 hour
        ))
        self._lock = threading.RLock()
        self._behavior_cache = AdvancedCache(CacheConfig(
            max_size=MAX_CACHE_SIZE,
            strategy=CacheStrategy.LRU,
            ttl=3600  # 1 hour
        ))
        self._behavior_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self._java_generators = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._behavior_locks = {}
        
        # Initialize template manager
        self.template_manager = TemplateManager(config)
    
    def _get_behavior_lock(self, behavior: str) -> threading.RLock:
        """Get or create a lock for a specific behavior."""
        with self._lock:
            if behavior not in self._behavior_locks:
                self._behavior_locks[behavior] = threading.RLock()
            return self._behavior_locks[behavior]
    
    @monitor_memory
    def generate_java_behaviors(self, analysis: AnalysisResult) -> str:
        """Generate Java code for behaviors that can't be handled by CSS."""
        try:
            # Check memory usage
            self._check_memory_usage()
            
            if not analysis.requires_java:
                return ""
            
            # Check behavior limit
            if len(analysis.requires_java) > MAX_JAVA_BEHAVIORS_PER_FILE:
                raise JavaBehaviorLimitException(
                    f"Too many Java behaviors: {len(analysis.requires_java)} > {MAX_JAVA_BEHAVIORS_PER_FILE}"
                )
            
            java = []
            
            # Add imports
            java.extend(f"import {imp};" for imp in JAVA_IMPORTS)
            java.append("")
            
            # Create behavior handler class
            java.append("public class BehaviorHandler {")
            
            # Generate behavior methods in parallel
            futures = []
            for behavior in analysis.requires_java:
                futures.append(self._java_generators.submit(self._generate_behavior_method, behavior))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=OPERATION_TIMEOUT)
                    java.extend(result)
                except Exception as e:
                    self.logger.error(f"Error generating behavior method: {str(e)}")
            
            java.append("}")
            
            return "\n".join(java)
        except Exception as e:
            self.logger.error(f"Error generating Java behaviors: {str(e)}")
            raise
    
    @monitor_memory
    def _check_memory_usage(self) -> None:
        """Check if memory usage exceeds limit."""
        process = psutil.Process()
        if process.memory_info().rss > MAX_MEMORY_USAGE:
            # Try to free memory
            gc.collect()
            if process.memory_info().rss > MAX_MEMORY_USAGE:
                raise MemoryLimitExceeded(
                    f"Memory usage exceeds limit: {process.memory_info().rss}"
                )
    
    def _generate_behavior_method(self, behavior: str) -> List[str]:
        """Generate Java method for a specific behavior."""
        behavior_lock = self._get_behavior_lock(behavior)
        with behavior_lock:
            try:
                # Check cache first
                cache_key = f"behavior:{behavior}"
                cached_result = self._cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
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
                self._cache.put(cache_key, result)
                
                return result
            except Exception as e:
                self.logger.error(f"Error generating behavior method: {str(e)}")
                return [f"    // Error generating {behavior} behavior: {str(e)}"]
    
    def behavior_to_java(self, behavior: Dict[str, Any]) -> str:
        """Convert behavior pattern to Java code."""
        behavior_lock = self._get_behavior_lock(behavior['type'])
        with behavior_lock:
            try:
                # Generate cache key
                cache_key = f"{behavior['type']}_{hash(str(behavior['context']))}"
                
                # Check cache first
                cached_code = self._behavior_cache.get(cache_key)
                if cached_code is not None:
                    return cached_code
                
                # Generate Java code
                template = self.template_manager.get_template('behaviors', behavior['type'])
                if not template:
                    return ""
                
                # Fill template
                code = template['base'].format(
                    class_name=behavior['class'],
                    method_name=behavior['method'],
                    context=behavior['context']
                )
                
                # Cache the result
                self._behavior_cache.put(cache_key, code)
                
                return code
            except Exception as e:
                self.logger.error(f"Error converting behavior to Java: {str(e)}")
                return ""
    
    def __del__(self):
        """Clean up resources with proper error handling."""
        try:
            self._cache.clear()
            self._behavior_cache.clear()
            self._java_generators.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

class ContentGenerator:
    """Generates optimized Java code by replacing style methods with CSS classes."""
    
    def __init__(self, config: Config):
        self.config = config
        self._cache = AdvancedCache(CacheConfig(
            max_size=MAX_CACHE_SIZE,
            strategy=CacheStrategy.LRU,
            ttl=3600  # 1 hour
        ))
        self._lock = threading.RLock()
        self._content_cache = AdvancedCache(CacheConfig(
            max_size=MAX_CACHE_SIZE,
            strategy=CacheStrategy.LRU,
            ttl=3600  # 1 hour
        ))
        self._content_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self._content_generators = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    @monitor_memory
    def generate_optimized_content(self, content: str, analysis: AnalysisResult) -> str:
        """Generate optimized Java content."""
        try:
            # Check memory usage
            self._check_memory_usage()
            
            # Generate efficient cache key
            cache_key = f"{hash(content)}_{id(analysis)}"
            
            # Check cache first
            cached_content = self._content_cache.get(cache_key)
            if cached_content is not None:
                return cached_content
            
            # Generate optimized content
            optimized = content
            
            # Get class mapping
            class_mapping = self._create_class_mapping(analysis)
            
            # Replace style methods with CSS classes in parallel
            futures = []
            for pattern in analysis.style_patterns:
                if pattern.css_property and pattern.css_value:
                    futures.append(
                        self._content_generators.submit(
                            self._replace_style_method,
                            optimized,
                            pattern,
                            class_mapping
                        )
                    )
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=OPERATION_TIMEOUT)
                    if result:
                        optimized = result
                except Exception as e:
                    self.logger.error(f"Error replacing style method: {str(e)}")
            
            # Update cache
            self._content_cache.put(cache_key, optimized)
            
            return optimized
        except Exception as e:
            self.logger.error(f"Error generating optimized content: {str(e)}")
            raise
    
    @monitor_memory
    def _check_memory_usage(self) -> None:
        """Check if memory usage exceeds limit."""
        process = psutil.Process()
        if process.memory_info().rss > MAX_MEMORY_USAGE:
            # Try to free memory
            gc.collect()
            if process.memory_info().rss > MAX_MEMORY_USAGE:
                raise MemoryLimitExceeded(
                    f"Memory usage exceeds limit: {process.memory_info().rss}"
                )
    
    def _replace_style_method(self, line: str, pattern: StylePattern, class_mapping: Dict[StylePattern, str]) -> str:
        """Replace a style method call with CSS class assignment."""
        try:
            # Find the style method call
            method_call = f"{pattern.method_name}({', '.join(pattern.arguments)})"
            
            # Replace with CSS class assignment
            if "setStyle" in line:
                # If there's already a setStyle call, add to it
                return re.sub(
                    r'setStyle\("([^"]*)"\)',
                    lambda m: f'setStyle("{m.group(1)} {class_mapping[pattern]}")',
                    line
                )
            else:
                # Otherwise, add a new setStyle call
                return line.replace(
                    method_call,
                    f'setStyle("{class_mapping[pattern]}")'
                )
        except Exception as e:
            self.logger.error(f"Error replacing style method: {str(e)}")
            return line
    
    def _create_class_mapping(self, analysis: AnalysisResult) -> Dict[StylePattern, str]:
        """Create mapping from patterns to CSS class names."""
        try:
            # Generate cache key
            cache_key = f"mapping_{id(analysis)}"
            
            # Check cache first
            cached_mapping = self._cache.get(cache_key)
            if cached_mapping is not None:
                return cached_mapping
            
            mapping = {}
            used_names = set()
            
            for pattern in analysis.style_patterns:
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
            
            # Cache the mapping
            self._cache.put(cache_key, mapping)
            
            return mapping
        except Exception as e:
            self.logger.error(f"Error creating class mapping: {str(e)}")
            return {}
    
    def __del__(self):
        """Clean up resources with proper error handling."""
        try:
            self._cache.clear()
            self._content_cache.clear()
            self._content_generators.shutdown(wait=True)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 