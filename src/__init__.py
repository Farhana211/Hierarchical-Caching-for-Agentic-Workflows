import hashlib
import json
from typing import Any, List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import logging
import threading

logger = logging.getLogger(__name__)


class WorkflowCache:
    
    def __init__(self, default_ttl: int = 300, dependency_graph=None):
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl
        self.dependency_graph = dependency_graph
        self.stats = {"workflow_hits": 0, "workflow_misses": 0}
        self._lock = threading.Lock()
    
    def _generate_workflow_key(
        self, 
        workflow: List[Tuple[str, str]], 
        session_id: Optional[str] = None
    ) -> str:

        workflow_str = json.dumps(workflow, sort_keys=True)
        key = hashlib.sha256(workflow_str.encode()).hexdigest()
        
        if session_id:
            key = f"session:{session_id}:{key}"
        
        return key
    
    def get_workflow(
        self, 
        workflow: List[Tuple[str, str]], 
        session_id: Optional[str] = None
    ) -> Optional[Any]:
        key = self._generate_workflow_key(workflow, session_id)
        
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if datetime.now() < entry["expires_at"]:
                    self.stats["workflow_hits"] += 1
                    logger.info(f"WORKFLOW CACHE HIT (saved {len(workflow)} tool calls)")
                    return entry["result"]
                else:
                    del self.cache[key]
            
            self.stats["workflow_misses"] += 1
        
        return None
    
    def cache_workflow(
        self, 
        workflow: List[Tuple[str, str]], 
        result: Any, 
        ttl: Optional[int] = None, 
        session_id: Optional[str] = None
    ):
        key = self._generate_workflow_key(workflow, session_id)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            self.cache[key] = {
                "workflow": workflow,
                "result": result,
                "cached_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=ttl)
            }
        
        logger.info(f"Cached workflow ({len(workflow)} steps, TTL: {ttl}s)")
    
    def invalidate_workflows_using_tool(
        self, 
        tool_name: str, 
        session_id: Optional[str] = None
    ):
        keys_to_delete = []

        dependent_tools = {tool_name}
        if self.dependency_graph:
            dependent_tools |= self.dependency_graph.get_invalidations(tool_name)
        
        with self._lock:
            for key, entry in self.cache.items():
                workflow = entry["workflow"]
                if any(step[0] in dependent_tools for step in workflow):
                    if session_id is None or key.startswith(f"session:{session_id}:"):
                        keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.cache[key]
        
        if keys_to_delete:
            logger.info(
                f"Invalidated {len(keys_to_delete)} workflows using "
                f"{tool_name} or its dependencies"
            )
    
    def clear(self):
        with self._lock:
            self.cache.clear()
            self.stats = {"workflow_hits": 0, "workflow_misses": 0}