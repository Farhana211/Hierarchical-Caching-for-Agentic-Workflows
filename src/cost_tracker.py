from typing import Dict, List
from datetime import datetime
import json
import logging
import threading

logger = logging.getLogger(__name__)


class CostTracker:
    
    def __init__(self, api_costs: Dict[str, float] = None):
        self.costs = {
            "total_api_calls": 0,
            "cached_calls": 0,
            "total_cost_usd": 0.0,
            "saved_cost_usd": 0.0,
            "tool_costs": {},
            "errors": {},
            "timeline": []
        }
        
        self.api_costs = api_costs or {
            "get_weather": 0.001,
            "get_user_location": 0.0005,
            "update_user_location": 0.002,
            "calculate_distance": 0.0001,
            "db_query_user": 0.0003,
            "db_query_products": 0.0005,
            "db_aggregation": 0.0008,
            "db_join_query": 0.001,
            "fs_read_file": 0.0001,
            "compute_fibonacci": 0.0002,
            "external_slow": 0.005,
        }
        
        self._lock = threading.Lock()
    
    def record_api_call(self, tool_name: str, was_cached: bool = False):
        cost = self.api_costs.get(tool_name, 0.001)
        
        with self._lock:
            self.costs["total_api_calls"] += 1
            
            if was_cached:
                self.costs["cached_calls"] += 1
                self.costs["saved_cost_usd"] += cost
            else:
                self.costs["total_cost_usd"] += cost
            
            if tool_name not in self.costs["tool_costs"]:
                self.costs["tool_costs"][tool_name] = {
                    "calls": 0,
                    "cached": 0,
                    "cost_usd": 0.0,
                    "saved_usd": 0.0
                }
            
            tool_stat = self.costs["tool_costs"][tool_name]
            tool_stat["calls"] += 1
            
            if was_cached:
                tool_stat["cached"] += 1
                tool_stat["saved_usd"] += cost
            else:
                tool_stat["cost_usd"] += cost
            
            self.costs["timeline"].append({
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "cached": was_cached,
                "cost": 0 if was_cached else cost
            })
    
    def record_error(self, tool_name: str, error_type: str):
        with self._lock:
            if tool_name not in self.costs["errors"]:
                self.costs["errors"][tool_name] = 0
            self.costs["errors"][tool_name] += 1
        logger.error(f"Error recorded for {tool_name}: {error_type}")
    
    def get_summary(self) -> Dict:
        with self._lock:
            total_calls = self.costs["total_api_calls"]
            cached_calls = self.costs["cached_calls"]
            cache_rate = (cached_calls / total_calls * 100) if total_calls > 0 else 0
            
            return {
                "total_api_calls": total_calls,
                "cached_calls": cached_calls,
                "cache_rate": f"{cache_rate:.1f}%",
                "total_cost_usd": f"${self.costs['total_cost_usd']:.4f}",
                "saved_cost_usd": f"${self.costs['saved_cost_usd']:.4f}",
                "error_count": sum(self.costs["errors"].values())
            }
    
    def get_tool_breakdown(self) -> Dict:
        with self._lock:
            breakdown = {}
            for tool_name, stats in self.costs["tool_costs"].items():
                cache_rate = (stats["cached"] / stats["calls"] * 100) if stats["calls"] > 0 else 0
                breakdown[tool_name] = {
                    "total_calls": stats["calls"],
                    "cached_calls": stats["cached"],
                    "cache_rate": f"{cache_rate:.1f}%",
                    "actual_cost": f"${stats['cost_usd']:.4f}",
                    "saved_cost": f"${stats['saved_usd']:.4f}",
                    "errors": self.costs["errors"].get(tool_name, 0)
                }
            return breakdown