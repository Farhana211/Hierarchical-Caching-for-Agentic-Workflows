import time
import json
import hashlib
import redis
from functools import lru_cache
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from collections import OrderedDict
import logging
import threading

logger = logging.getLogger(__name__)


class NoCacheSystem:
    def __init__(self):
        self.stats = {"hits": 0, "misses": 0, "total_calls": 0}
        self._lock = threading.Lock()
        
        from cost_tracker import CostTracker
        self.cost_tracker = CostTracker()
    
    def get(self, tool_name: str, params: Dict, session_id: str = None) -> Optional[Any]:
        with self._lock:
            self.stats["misses"] += 1
            self.stats["total_calls"] += 1
        
        self.cost_tracker.record_api_call(tool_name, was_cached=False)
        return None
    
    def set(self, tool_name: str, params: Dict, result: Any, session_id: str = None, ttl: Optional[int] = None):
        pass
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                **self.stats,
                "hit_rate": "0.0%",
                "cache_size": 0,
                "system": "NoCache"
            }
    def get_cost_summary(self) -> Dict:
        """Expose cost tracker summary"""
        return self.cost_tracker.get_summary()

    def get_cost_breakdown(self) -> Dict:
        """Expose cost tracker breakdown"""
        return self.cost_tracker.get_tool_breakdown()
    
    def clear(self):
        with self._lock:
            self.stats = {"hits": 0, "misses": 0, "total_calls": 0}
    
    def invalidate_dependencies(self, tool_name: str, session_id: str = None):
        pass
    
class SimpleMemoizationCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.stats = {"hits": 0, "misses": 0, "total_calls": 0}
        self._lock = threading.Lock()
        
        from cost_tracker import CostTracker
        self.cost_tracker = CostTracker()
    
    def _generate_key(self, tool_name: str, params: Dict, session_id: str = None) -> str:
        params_str = json.dumps(params, sort_keys=True)
        key = hashlib.sha256(f"{tool_name}:{params_str}".encode()).hexdigest()
        if session_id:
            key = f"{session_id}:{key}"
        return key
    
    def get(self, tool_name: str, params: Dict, session_id: str = None) -> Optional[Any]:
        key = self._generate_key(tool_name, params, session_id)
        
        with self._lock:
            self.stats["total_calls"] += 1
            
            if key in self.cache:
                self.stats["hits"] += 1
                self.cost_tracker.record_api_call(tool_name, was_cached=True)
                return self.cache[key]
            
            self.stats["misses"] += 1
        
        self.cost_tracker.record_api_call(tool_name, was_cached=False)
        return None
    
    def set(self, tool_name: str, params: Dict, result: Any, session_id: str = None, ttl: Optional[int] = None):
        key = self._generate_key(tool_name, params, session_id)
        with self._lock:
            self.cache[key] = result
    
    def get_stats(self) -> Dict:
        with self._lock:
            total = self.stats["total_calls"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            return {
                **self.stats,
                "hit_rate": f"{hit_rate:.2f}%",
                "cache_size": len(self.cache),
                "system": "SimpleMemoization"
            }
    
    
    def get_cost_summary(self) -> Dict:
        """Expose cost tracker summary"""
        return self.cost_tracker.get_summary()

    def get_cost_breakdown(self) -> Dict:
        """Expose cost tracker breakdown"""
        return self.cost_tracker.get_tool_breakdown()
    
    def clear(self):
        with self._lock:
            self.cache.clear()
            self.stats = {"hits": 0, "misses": 0, "total_calls": 0}
    
    def invalidate_dependencies(self, tool_name: str, session_id: str = None):
        pass
    


class LRUCache:
    def __init__(self, maxsize: int = 128):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_calls": 0}
        self._lock = threading.Lock()
        
        from cost_tracker import CostTracker
        self.cost_tracker = CostTracker()
    
    def _generate_key(self, tool_name: str, params: Dict, session_id: str = None) -> str:
        params_str = json.dumps(params, sort_keys=True)
        key = hashlib.sha256(f"{tool_name}:{params_str}".encode()).hexdigest()
        if session_id:
            key = f"{session_id}:{key}"
        return key
    
    def get(self, tool_name: str, params: Dict, session_id: str = None) -> Optional[Any]:
        key = self._generate_key(tool_name, params, session_id)
        
        with self._lock:
            self.stats["total_calls"] += 1
            
            if key in self.cache:
                self.stats["hits"] += 1
                self.cache.move_to_end(key)
                self.cost_tracker.record_api_call(tool_name, was_cached=True)
                return self.cache[key]
            
            self.stats["misses"] += 1
        
        self.cost_tracker.record_api_call(tool_name, was_cached=False)
        return None
    
    def set(self, tool_name: str, params: Dict, result: Any, session_id: str = None, ttl: Optional[int] = None):
        key = self._generate_key(tool_name, params, session_id)
        
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = result
            else:
                self.cache[key] = result
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
                    self.stats["evictions"] += 1
    
    def get_stats(self) -> Dict:
        with self._lock:
            total = self.stats["total_calls"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            return {
                **self.stats,
                "hit_rate": f"{hit_rate:.2f}%",
                "cache_size": len(self.cache),
                "maxsize": self.maxsize,
                "system": f"LRU_{self.maxsize}"
            }
    
    def get_cost_summary(self) -> Dict:
        """Expose cost tracker summary"""
        return self.cost_tracker.get_summary()

    def get_cost_breakdown(self) -> Dict:
        """Expose cost tracker breakdown"""
        return self.cost_tracker.get_tool_breakdown()
    
    def clear(self):
        with self._lock:
            self.cache.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_calls": 0}
    
    def invalidate_dependencies(self, tool_name: str, session_id: str = None):
        pass


class TTLOnlyCache:
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl
        self.stats = {"hits": 0, "misses": 0, "expirations": 0, "total_calls": 0}
        self._lock = threading.Lock()
        
        from cost_tracker import CostTracker
        self.cost_tracker = CostTracker()
    
    def _generate_key(self, tool_name: str, params: Dict, session_id: str = None) -> str:
        params_str = json.dumps(params, sort_keys=True)
        key = hashlib.sha256(f"{tool_name}:{params_str}".encode()).hexdigest()
        if session_id:
            key = f"{session_id}:{key}"
        return key
    
    def get(self, tool_name: str, params: Dict, session_id: str = None) -> Optional[Any]:
        key = self._generate_key(tool_name, params, session_id)
        
        with self._lock:
            self.stats["total_calls"] += 1
            
            if key in self.cache:
                entry = self.cache[key]
                if datetime.now() < entry["expires_at"]:
                    self.stats["hits"] += 1
                    self.cost_tracker.record_api_call(tool_name, was_cached=True)
                    return entry["result"]
                else:
                    del self.cache[key]
                    self.stats["expirations"] += 1
            
            self.stats["misses"] += 1
        
        self.cost_tracker.record_api_call(tool_name, was_cached=False)
        return None
    
    def set(self, tool_name: str, params: Dict, result: Any, session_id: str = None, ttl: Optional[int] = None):
        key = self._generate_key(tool_name, params, session_id)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            self.cache[key] = {
                "result": result,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=ttl)
            }
    
    def get_stats(self) -> Dict:
        with self._lock:
            total = self.stats["total_calls"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            return {
                **self.stats,
                "hit_rate": f"{hit_rate:.2f}%",
                "cache_size": len(self.cache),
                "system": f"TTL_{self.default_ttl}s"
            }
    def get_cost_summary(self) -> Dict:
        """Expose cost tracker summary"""
        return self.cost_tracker.get_summary()

    def get_cost_breakdown(self) -> Dict:
        """Expose cost tracker breakdown"""
        return self.cost_tracker.get_tool_breakdown()
    
    def clear(self):
        with self._lock:
            self.cache.clear()
            self.stats = {"hits": 0, "misses": 0, "expirations": 0, "total_calls": 0}
    
    def invalidate_dependencies(self, tool_name: str, session_id: str = None):
        pass


class RawRedisCache:
    def __init__(self, host='localhost', port=6379, db=2, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self.stats = {"hits": 0, "misses": 0, "redis_errors": 0, "total_calls": 0}
        self.fallback_cache = {}
        self._lock = threading.Lock()
        
        from cost_tracker import CostTracker
        self.cost_tracker = CostTracker()
        
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.redis_client.ping()
            self.using_redis = True
            logger.info(f"Raw Redis connected at {host}:{port}/db{db}")
        except:
            logger.warning("Redis unavailable for raw_redis, using fallback")
            self.using_redis = False
            self.redis_client = None
    
    def _generate_key(self, tool_name: str, params: Dict, session_id: str = None) -> str:
        params_str = json.dumps(params, sort_keys=True)
        key = f"raw_redis:{tool_name}:{hashlib.sha256(params_str.encode()).hexdigest()}"
        if session_id:
            key = f"{session_id}:{key}"
        return key
    
    def get(self, tool_name: str, params: Dict, session_id: str = None) -> Optional[Any]:
        key = self._generate_key(tool_name, params, session_id)
        
        with self._lock:
            self.stats["total_calls"] += 1
        
        if self.using_redis:
            try:
                data = self.redis_client.get(key)
                if data:
                    with self._lock:
                        self.stats["hits"] += 1
                    self.cost_tracker.record_api_call(tool_name, was_cached=True)
                    return json.loads(data.decode())
            except Exception as e:
                with self._lock:
                    self.stats["redis_errors"] += 1
        else:
            with self._lock:
                if key in self.fallback_cache:
                    entry = self.fallback_cache[key]
                    if datetime.now() < entry["expires_at"]:
                        self.stats["hits"] += 1
                        self.cost_tracker.record_api_call(tool_name, was_cached=True)
                        return entry["result"]
                    else:
                        del self.fallback_cache[key]
        
        with self._lock:
            self.stats["misses"] += 1
        
        self.cost_tracker.record_api_call(tool_name, was_cached=False)
        return None
    
    def set(self, tool_name: str, params: Dict, result: Any, session_id: str = None, ttl: Optional[int] = None):
        key = self._generate_key(tool_name, params, session_id)
        ttl = ttl or self.default_ttl
        
        if self.using_redis:
            try:
                serialized = json.dumps(result, default=str).encode()
                self.redis_client.setex(key, ttl, serialized)
            except:
                pass
        else:
            with self._lock:
                self.fallback_cache[key] = {
                    "result": result,
                    "expires_at": datetime.now() + timedelta(seconds=ttl)
                }
    
    def get_stats(self) -> Dict:
        with self._lock:
            total = self.stats["total_calls"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            cache_size = len(self.fallback_cache)
        
        if self.using_redis and self.redis_client:
            try:
                cache_size = self.redis_client.dbsize()
            except:
                pass
        
        return {
            **self.stats,
            "hit_rate": f"{hit_rate:.2f}%",
            "cache_size": cache_size,
            "system": "RawRedis"
        }
    
    def get_cost_summary(self) -> Dict:
        """Expose cost tracker summary"""
        return self.cost_tracker.get_summary()

    def get_cost_breakdown(self) -> Dict:
        """Expose cost tracker breakdown"""
        return self.cost_tracker.get_tool_breakdown()
    
    def clear(self):
        if self.using_redis and self.redis_client:
            try:
                self.redis_client.flushdb()
            except:
                pass
        with self._lock:
            self.fallback_cache.clear()
            self.stats = {"hits": 0, "misses": 0, "redis_errors": 0, "total_calls": 0}
    
    def invalidate_dependencies(self, tool_name: str, session_id: str = None):
        pass


def get_all_baseline_systems():
    return {
        "no_cache": NoCacheSystem(),
        "simple_memoization": SimpleMemoizationCache(),
        "lru_128": LRUCache(maxsize=128),
        "lru_512": LRUCache(maxsize=512),
        "ttl_300s": TTLOnlyCache(default_ttl=300),
        "raw_redis": RawRedisCache(default_ttl=300)
    }