import hashlib
import json
import time
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from tools import TOOLS
from dependency_graph import ToolDependencyGraph
from redis_cache import RedisCache
from cost_tracker import CostTracker
import logging
import threading

logger = logging.getLogger(__name__)


class DependencyAwareCache:
    def __init__(
        self, 
        adaptive_ttl_manager=None, 
        use_redis=True, 
        redis_host='localhost', 
        redis_port=6379,
        staleness_threshold=0.3
    ):
        self.use_redis = use_redis
        if use_redis:
            self.redis_cache = RedisCache(
                host=redis_host, 
                port=redis_port, 
                fallback_to_memory=False,
                fail_fast=True
            )
            self.cache = None
        else:
            self.cache: Dict[str, Dict] = {}
            self.redis_cache = None
        
        self.dependency_graph = ToolDependencyGraph()
        self.adaptive_ttl = adaptive_ttl_manager
        self.cost_tracker = CostTracker()
        self.staleness_threshold = staleness_threshold
        
        self._thread_local = threading.local()
        
        self.redis_metadata: Dict[str, Dict] = {}
        self._metadata_lock = threading.Lock()
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "dependency_invalidations": 0,
            "stale_hits": 0,
            "expired_hits": 0,
            "redis_stale_detections": 0,
            "memory_stale_detections": 0
        }
        
        self._lock = threading.RLock()
        self._lock_timeout = 30
        
        logger.info(
            f"DependencyAwareCache initialized: staleness_threshold={staleness_threshold:.1%}, "
            f"Redis staleness detection={'ENABLED' if use_redis else 'N/A'}"
        )
    
    def _generate_key(self, tool_name: str, params: Dict, session_id: Optional[str] = None) -> str:
        if self.use_redis and self.redis_cache:
            return self.redis_cache._generate_key(tool_name, params, session_id)
        else:
            params_str = json.dumps(params, sort_keys=True)
            key = hashlib.sha256(f"{tool_name}:{params_str}".encode()).hexdigest()
            if session_id:
                key = f"session:{session_id}:{key}"
            return key
    
    def _calculate_redis_staleness(self, key: str) -> bool:
        with self._metadata_lock:
            if key not in self.redis_metadata:
                return False 
            
            metadata = self.redis_metadata[key]
            insertion_time = metadata["inserted_at"]
            original_ttl = metadata["ttl"]
        
        if original_ttl <= 0:
            return False
        
        age_seconds = (datetime.now() - insertion_time).total_seconds()
        age_ratio = age_seconds / original_ttl
        is_stale = age_ratio > self.staleness_threshold
        
        if is_stale:
            with self._lock:
                self.stats["redis_stale_detections"] += 1
            logger.debug(
                f"Redis staleness detected: age={age_seconds:.0f}s, "
                f"original_ttl={original_ttl}s ({age_ratio:.1%})"
            )
        
        return is_stale
    
    def _record_redis_metadata(self, key: str, ttl: int):
        with self._metadata_lock:
            self.redis_metadata[key] = {
                "inserted_at": datetime.now(),
                "ttl": ttl
            }

            if len(self.redis_metadata) > 5000:
                if not hasattr(self, '_cleanup_counter'):
                    self._cleanup_counter = 0
                self._cleanup_counter += 1
                
                if self._cleanup_counter >= 20:
                    self._cleanup_counter = 0
                    max_ttl = getattr(self.adaptive_ttl, 'max_ttl', 3600) if self.adaptive_ttl else 3600
                    cutoff = datetime.now() - timedelta(seconds=max_ttl * 2)
                    old_size = len(self.redis_metadata)
                    self.redis_metadata = {
                        k: v for k, v in self.redis_metadata.items()
                        if v["inserted_at"] > cutoff
                    }
                    new_size = len(self.redis_metadata)
                    if old_size - new_size > 100:
                        logger.debug(f"Cleaned up {old_size - new_size} old Redis metadata entries, now tracking {new_size}")
    
    def get(
        self, 
        tool_name: str, 
        params: Dict, 
        session_id: Optional[str] = None
    ) -> Optional[Any]:

        key = self._generate_key(tool_name, params, session_id)
        
        cached_result = None
        is_stale = False
        is_expired = False
        
        self._thread_local.last_staleness_info = False
        
        if self.use_redis and self.redis_cache:
            try:
                cached_result = self.redis_cache.get(tool_name, params, session_id)
                
                if cached_result is not None:
                    is_stale = self._calculate_redis_staleness(key)
                    self._thread_local.last_staleness_info = is_stale
                    
                    if is_stale:
                        with self._lock:
                            self.stats["stale_hits"] += 1
                        logger.debug(f"Redis stale hit for {tool_name}")
                
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                cached_result = None
        else:
            with self._lock:
                if key in self.cache:
                    entry = self.cache[key]
                    now = datetime.now()
                    
                    if now >= entry["expires_at"]:
                        is_expired = True
                        del self.cache[key]
                        self.stats["expired_hits"] += 1
                    else:
                        age_seconds = (now - entry["cached_at"]).total_seconds()
                        ttl_seconds = entry.get("ttl", 300)
                        age_ratio = age_seconds / ttl_seconds if ttl_seconds > 0 else 0
                        
                        if age_ratio > self.staleness_threshold:
                            is_stale = True
                            self._thread_local.last_staleness_info = True
                            self.stats["stale_hits"] += 1
                            self.stats["memory_stale_detections"] += 1
                            
                            logger.debug(
                                f"Memory stale hit for {tool_name}: "
                                f"age={age_seconds:.0f}s, ttl={ttl_seconds:.0f}s ({age_ratio:.1%})"
                            )
                        
                        cached_result = entry["result"]
        
        if is_expired:
            with self._lock:
                self.stats["misses"] += 1
                self.stats["invalidations"] += 1
            
            self.cost_tracker.record_api_call(tool_name, was_cached=False)
            
            if self.adaptive_ttl:
                self.adaptive_ttl.record_access(tool_name, was_hit=False, was_stale=False)
            
            logger.debug(f"Cache EXPIRED: {tool_name}")
            return None
        
        elif cached_result is not None:
            with self._lock:
                self.stats["hits"] += 1
            
            self.cost_tracker.record_api_call(tool_name, was_cached=True)
            
            if self.adaptive_ttl:
                self.adaptive_ttl.record_access(tool_name, was_hit=True, was_stale=is_stale)
            
            hit_type = "STALE" if is_stale else "FRESH"
            logger.debug(f"Cache HIT ({hit_type}): {tool_name}")
            return cached_result
        
        else:
            with self._lock:
                self.stats["misses"] += 1
            
            self.cost_tracker.record_api_call(tool_name, was_cached=False)
            
            if self.adaptive_ttl:
                self.adaptive_ttl.record_access(tool_name, was_hit=False, was_stale=False)
            
            logger.debug(f"Cache MISS: {tool_name}")
            return None
    
    def set(
        self, 
        tool_name: str, 
        params: Dict, 
        result: Any, 
        session_id: Optional[str] = None,
        ttl: Optional[int] = None
    ):
  
        if tool_name not in TOOLS:
            logger.warning(f"Unknown tool: {tool_name}, not caching")
            return
        
        tool = TOOLS[tool_name]
        
        if not tool.deterministic:
            logger.info(f"Skipping cache for non-deterministic tool: {tool_name}")
            return
        
        base_ttl = tool.base_ttl
        if ttl is None:
            if self.adaptive_ttl:
                ttl = self.adaptive_ttl.get_recommended_ttl(tool_name, base_ttl)
                if ttl != base_ttl:
                    logger.debug(f"Adaptive TTL for {tool_name}: {base_ttl}s → {ttl}s")
            else:
                ttl = base_ttl
        
        key = self._generate_key(tool_name, params, session_id)
        
        if self.use_redis and self.redis_cache:
            try:
                self.redis_cache.set(tool_name, params, result, ttl, session_id)
                self._record_redis_metadata(key, ttl)
                logger.debug(f"Cached (Redis): {tool_name} (TTL: {ttl}s)")
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        else:
            with self._lock:
                self.cache[key] = {
                    "tool_name": tool_name,
                    "params": params,
                    "result": result,
                    "cached_at": datetime.now(),
                    "expires_at": datetime.now() + timedelta(seconds=ttl),
                    "ttl": ttl
                }
            logger.debug(f"Cached: {tool_name} (TTL: {ttl}s)")
    
    def invalidate_dependencies(
        self, 
        tool_name: str, 
        session_id: Optional[str] = None
    ):

        tools_to_invalidate = self.dependency_graph.get_invalidations(tool_name)
        
        if not tools_to_invalidate:
            return
        
        logger.info(f"Invalidating dependencies of {tool_name}: {tools_to_invalidate}")
        
        total_invalidated = 0
        
        for dependent_tool in tools_to_invalidate:
            count = 0
            
            if self.use_redis and self.redis_cache:
                pattern = f"session:{session_id}:tool:{dependent_tool}:*" if session_id else f"tool:{dependent_tool}:*"
                try:
                    if self.redis_cache.using_redis:
                        keys = self.redis_cache.redis_client.keys(pattern)
                        count = len(keys)
                        if keys:
                            self.redis_cache.redis_client.delete(*keys)
                            with self._metadata_lock:
                                for key in keys:
                                    key_str = key.decode() if isinstance(key, bytes) else key
                                    self.redis_metadata.pop(key_str, None)
                except Exception as e:
                    logger.error(f"Redis invalidation error: {e}")
            else:
                with self._lock:
                    keys_to_delete = []
                    for key, entry in self.cache.items():
                        if entry["tool_name"] == dependent_tool:
                            if session_id is None or key.startswith(f"session:{session_id}:"):
                                keys_to_delete.append(key)
                                count += 1
                    
                    for key in keys_to_delete:
                        del self.cache[key]
            
            if count > 0:
                total_invalidated += count
                logger.info(f"Invalidated {count} cached entries for {dependent_tool}")
        
        if total_invalidated > 0:
            with self._lock:
                self.stats["dependency_invalidations"] += total_invalidated
    
    def get_stats(self) -> Dict:
        with self._lock:
            stats_snapshot = dict(self.stats)
            total = stats_snapshot["hits"] + stats_snapshot["misses"]
            hit_rate = (stats_snapshot["hits"] / total * 100) if total > 0 else 0
            staleness_rate = (stats_snapshot["stale_hits"] / stats_snapshot["hits"] * 100) if stats_snapshot["hits"] > 0 else 0
        
        cache_size = 0
        if self.use_redis and self.redis_cache:
            try:
                cache_size = self.redis_cache.get_cache_size()
            except:
                cache_size = 0
        else:
            with self._lock:
                cache_size = len(self.cache) if self.cache else 0
        
        with self._metadata_lock:
            metadata_tracked = len(self.redis_metadata)
        
        return {
            **stats_snapshot,
            "hit_rate": f"{hit_rate:.2f}%",
            "staleness_rate": f"{staleness_rate:.2f}%",
            "cache_size": cache_size,
            "backend": "Redis" if self.use_redis else "In-Memory",
            "redis_metadata_tracked": metadata_tracked if self.use_redis else 0,
            "staleness_threshold": f"{self.staleness_threshold:.1%}"
        }
    
    def get_cost_summary(self) -> Dict:
        return self.cost_tracker.get_summary()
    
    def get_cost_breakdown(self) -> Dict:
        return self.cost_tracker.get_tool_breakdown()
    
    def clear(self):
        if self.use_redis and self.redis_cache:
            try:
                self.redis_cache.clear_all()
                with self._metadata_lock:
                    self.redis_metadata.clear()
            except:
                pass
        else:
            with self._lock:
                if self.cache:
                    self.cache.clear()
        
        with self._lock:
            self.stats = {
                "hits": 0,
                "misses": 0,
                "invalidations": 0,
                "dependency_invalidations": 0,
                "stale_hits": 0,
                "expired_hits": 0,
                "redis_stale_detections": 0,
                "memory_stale_detections": 0
            }
    
    def get_staleness_info(self) -> Dict:
        staleness_detected = getattr(self._thread_local, 'last_staleness_info', False)
        
        with self._metadata_lock:
            metadata_count = len(self.redis_metadata)
        
        return {
            "last_staleness_detected": staleness_detected,
            "redis_metadata_tracked": metadata_count,
            "staleness_threshold": self.staleness_threshold,
            "total_stale_hits": self.stats["stale_hits"],
            "redis_stale_detections": self.stats["redis_stale_detections"],
            "memory_stale_detections": self.stats["memory_stale_detections"]
        }