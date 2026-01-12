import os
import redis
import json
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import logging
import threading

logger = logging.getLogger(__name__)


class RedisCache:
   
    def __init__(
        self, 
        host='localhost', 
        port=6379, 
        db=0, 
        fallback_to_memory=False,
        fail_fast=True
    ):
        db = int(os.getenv('REDIS_DB', db)) 
        
        self.fallback_to_memory = fallback_to_memory
        self.fail_fast = fail_fast
        self.memory_cache = {}
        self._lock = threading.Lock()
        
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db,
                decode_responses=False,
                socket_connect_timeout=10,
                socket_timeout=30,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self.redis_client.ping()
            self.using_redis = True
            logger.info(f"Connected to Redis at {host}:{port}/db{db}")
        except (redis.ConnectionError, redis.RedisError) as e:
            if fail_fast:
                logger.error(f"Redis connection failed in fail-fast mode: {e}")
                raise RuntimeError(
                    f"CRITICAL: Redis unavailable at {host}:{port}. "
                    f"Cannot run evaluation without Redis. "
                    f"Please start Redis with: redis-server --port {port}"
                )
            elif fallback_to_memory:
                logger.warning(f"Redis unavailable, falling back to memory: {e}")
                self.using_redis = False
                self.redis_client = None
            else:
                raise
    
    def _serialize(self, obj: Any) -> bytes:
        """Serialize Python object for storage"""
        return json.dumps(obj, default=str).encode()
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize stored data"""
        return json.loads(data.decode())
    
    def _generate_key(
        self, 
        tool_name: str, 
        params: Dict, 
        session_id: Optional[str] = None
    ) -> str:
        """Generate cache key with session scoping"""
        params_str = json.dumps(params, sort_keys=True)
        key = f"tool:{tool_name}:{hashlib.sha256(params_str.encode()).hexdigest()}"
        if session_id:
            key = f"session:{session_id}:{key}"
        return key
    
    def get(
        self, 
        tool_name: str, 
        params: Dict, 
        session_id: Optional[str] = None
    ) -> Optional[Any]:
        """Get cached result with session scoping"""
        key = self._generate_key(tool_name, params, session_id)
        
        if self.using_redis:
            try:
                data = self.redis_client.get(key)
                if data:
                    return self._deserialize(data)
            except redis.RedisError as e:
                logger.error(f"Redis get error: {e}")
                if self.fail_fast:
                    raise
                return None
        else:
            with self._lock:
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    if datetime.now() < entry["expires_at"]:
                        return entry["result"]
                    else:
                        del self.memory_cache[key]
        
        return None
    
    def set(
        self, 
        tool_name: str, 
        params: Dict, 
        result: Any, 
        ttl: int, 
        session_id: Optional[str] = None
    ):
        key = self._generate_key(tool_name, params, session_id)
        
        if self.using_redis:
            try:
                serialized = self._serialize(result)
                self.redis_client.setex(key, ttl, serialized)
            except redis.RedisError as e:
                logger.error(f"Redis set error: {e}")
                if self.fail_fast:
                    raise
        else:
            with self._lock:
                self.memory_cache[key] = {
                    "result": result,
                    "expires_at": datetime.now() + timedelta(seconds=ttl)
                }
    
    def delete(
        self, 
        tool_name: str, 
        params: Dict, 
        session_id: Optional[str] = None
    ):
        key = self._generate_key(tool_name, params, session_id)
        
        if self.using_redis:
            try:
                self.redis_client.delete(key)
            except redis.RedisError as e:
                logger.error(f"Redis delete error: {e}")
                if self.fail_fast:
                    raise
        else:
            with self._lock:
                self.memory_cache.pop(key, None)
    
    def delete_pattern(self, pattern: str):
        if self.using_redis:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except redis.RedisError as e:
                logger.error(f"Redis delete_pattern error: {e}")
                if self.fail_fast:
                    raise
        else:
            with self._lock:
                pattern_str = pattern.replace('*', '')
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern_str in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
    
    def get_cache_size(self) -> int:
        if self.using_redis:
            try:
                return self.redis_client.dbsize()
            except redis.RedisError as e:
                logger.error(f"Redis dbsize error: {e}")
                if self.fail_fast:
                    raise
                return 0
        else:
            with self._lock:
                return len(self.memory_cache)
    
    def clear_all(self):
        if self.using_redis:
            try:
                self.redis_client.flushdb()
            except redis.RedisError as e:
                logger.error(f"Redis flushdb error: {e}")
                if self.fail_fast:
                    raise
        else:
            with self._lock:
                self.memory_cache.clear()
    
    def health_check(self) -> bool:
        if self.using_redis:
            try:
                import socket
                socket.setdefaulttimeout(5.0)
                self.redis_client.ping()
                return True
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.error(f"Redis health check failed: {e}")
                try:
                    self.redis_client.connection_pool.disconnect()
                    self.redis_client = redis.Redis( 
                        db=self.db,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True
                    )
                    self.redis_client.ping()
                    return True
                except:
                    return False
        return True