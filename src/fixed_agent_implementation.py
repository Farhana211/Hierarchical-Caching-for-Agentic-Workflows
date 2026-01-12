import json
import time
import logging
import re
import threading
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from cache_with_dependencies import DependencyAwareCache
from adaptive_ttl import AdaptiveTTLManager
from workflow_cache import WorkflowCache
from tools import TOOLS

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    tool_calls_attempted: int = 0
    tool_calls_executed: int = 0
    tool_cache_hits: int = 0
    workflow_queries: int = 0
    workflow_cache_hits: int = 0
    tools_saved_by_workflow: int = 0
    execution_time_saved: float = 0.0
    api_calls_saved: int = 0
    
    def get_tool_hit_rate(self) -> float:
        if self.tool_calls_attempted == 0:
            return 0.0
        return (self.tool_cache_hits / self.tool_calls_attempted) * 100
    
    def get_workflow_hit_rate(self) -> float:
        if self.workflow_queries == 0:
            return 0.0
        return (self.workflow_cache_hits / self.workflow_queries) * 100
    
    def get_overall_efficiency(self) -> float:
        total_possible = self.tool_calls_attempted + self.tools_saved_by_workflow
        if total_possible == 0:
            return 0.0
        total_saved = self.tool_cache_hits + self.tools_saved_by_workflow
        return (total_saved / total_possible) * 100


class ImprovedToolParser:
    
    def __init__(self):
        self.known_cities = {
            "new york", "london", "tokyo", "paris", "berlin", "sydney",
            "toronto", "moscow", "beijing", "mumbai", "dubai", "singapore",
            "los angeles", "chicago", "san francisco", "boston", "seattle",
            "miami", "houston", "atlanta", "madrid", "rome", "barcelona",
            "amsterdam", "stockholm", "copenhagen", "vienna", "prague",
            "warsaw", "budapest", "istanbul", "cairo", "rio", "mexico city"
        }
        self._lock = threading.Lock()
    
    def extract_cities(self, text: str) -> List[str]:
        with self._lock:
            text_lower = text.lower()
            cities_found = []
            
            for city in self.known_cities:
                if city in text_lower:
                    cities_found.append(city.title())
            
            return list(dict.fromkeys(cities_found))
    
    def extract_user_ids(self, text: str) -> List[int]:
        matches = re.findall(r'user[_\s]?(?:id)?[:\s]*(\d+)', text.lower())
        return [int(m) for m in matches]
    
    def extract_numbers(self, text: str) -> List[int]:
        matches = re.findall(r'\b(\d+)\b', text)
        return [int(m) for m in matches if int(m) < 1000]
    
    def extract_categories(self, text: str) -> List[str]:
        categories = ["electronics", "clothing", "food", "books", "home"]
        found = []
        text_lower = text.lower()
        for cat in categories:
            if cat in text_lower:
                found.append(cat.title())
        return found
    
    def extract_departments(self, text: str) -> List[str]:
        departments = ["engineering", "sales", "marketing", "support", "finance", "hr"]
        found = []
        text_lower = text.lower()
        for dept in departments:
            if dept in text_lower:
                found.append(dept.title())
        return found
    
    def parse_tool_calls(self, question: str, llm_response: str = None) -> List[Dict]:
        tool_calls = []
        full_text = question
        if llm_response:
            full_text += " " + llm_response
        
        text_lower = full_text.lower()
        
        if any(word in text_lower for word in ["update", "set", "move"]) and "location" in text_lower:
            user_ids = self.extract_user_ids(full_text)
            cities = self.extract_cities(full_text)
            if user_ids and cities:
                tool_calls.append({
                    "tool": "update_user_location",
                    "arguments": {"user_id": user_ids[0], "city": cities[0], "lat": 40.0, "lon": -74.0}
                })

        if any(word in text_lower for word in ["order", "purchase", "buy", "create order"]) and "user" in text_lower and "product" in text_lower:
            user_ids = self.extract_user_ids(full_text)
            numbers = self.extract_numbers(full_text)
            if user_ids and len(numbers) >= 2:
                tool_calls.append({
                    "tool": "db_write_order",
                    "arguments": {"user_id": user_ids[0], "product_id": numbers[0], "quantity": numbers[1]}
                })

        if any(word in text_lower for word in ["write", "save", "update"]) and "file" in text_lower:
            filenames = re.findall(r'file[:\s]*(\w+\.\w+)', text_lower)
            if filenames:
                content_match = re.search(r'content[:\s]*([^\.]+)', text_lower)
                content = content_match.group(1).strip() if content_match else "default content"
                tool_calls.append({
                    "tool": "fs_write_file",
                    "arguments": {"filename": filenames[0], "content": content}
                })

        if any(word in text_lower for word in ["weather", "temperature", "forecast", "rain", "snow"]):
            cities = self.extract_cities(full_text)
            for city in cities:
                tool_calls.append({
                    "tool": "get_weather",
                    "arguments": {"city": city}
                })

        user_ids = self.extract_user_ids(full_text)
        if ("location" in text_lower or "where" in text_lower) and user_ids:
            for user_id in user_ids:
                tool_calls.append({
                    "tool": "get_user_location",
                    "arguments": {"user_id": user_id}
                })

        if "distance" in text_lower and len(user_ids) >= 2:
            tool_calls.append({
                "tool": "calculate_distance",
                "arguments": {"needs_locations": user_ids[:2]}
            })
        
        if "user" in text_lower and ("profile" in text_lower or "information" in text_lower or "details" in text_lower):
            for user_id in user_ids:
                tool_calls.append({
                    "tool": "db_query_user",
                    "arguments": {"user_id": user_id}
                })
        
        categories = self.extract_categories(full_text)
        if categories and ("product" in text_lower or "items" in text_lower):
            for category in categories:
                tool_calls.append({
                    "tool": "db_query_products",
                    "arguments": {"category": category}
                })
        
        departments = self.extract_departments(full_text)
        if departments and ("statistics" in text_lower or "stats" in text_lower or "summary" in text_lower):
            for dept in departments:
                tool_calls.append({
                    "tool": "db_aggregation",
                    "arguments": {"department": dept}
                })
        
        if user_ids and ("order" in text_lower or "purchase" in text_lower) and "history" in text_lower:
            for user_id in user_ids:
                tool_calls.append({
                    "tool": "db_join_query",
                    "arguments": {"user_id": user_id}
                })

        if "file" in text_lower or "read" in text_lower:
            filenames = re.findall(r'(\w+\.\w+)', full_text)
            for filename in filenames:
                tool_calls.append({
                    "tool": "fs_read_file",
                    "arguments": {"filename": filename}
                })
        
        if "config" in text_lower:
            config_names = re.findall(r'(\w+)(?:\s+config)', text_lower)
            for config_name in config_names:
                tool_calls.append({
                    "tool": "fs_read_config",
                    "arguments": {"config_name": config_name}
                })
        if "fibonacci" in text_lower:
            numbers = self.extract_numbers(full_text)
            if numbers:
                tool_calls.append({
                    "tool": "compute_fibonacci",
                    "arguments": {"n": numbers[0]}
                })
        
        if "statistics" in text_lower or "stats" in text_lower:
            number_list = self.extract_numbers(full_text)
            if len(number_list) > 2:
                tool_calls.append({
                    "tool": "compute_statistics",
                    "arguments": {"numbers": number_list[:10]}
                })

        if "external" in text_lower or "search" in text_lower:
            query_terms = re.findall(r'(?:for|search|query)\s+(\w+)', text_lower)
            if query_terms:
                if "fast" in text_lower:
                    api_type = "external_fast"
                elif "slow" in text_lower:
                    api_type = "external_slow"
                else:
                    api_type = "external_medium"
                
                tool_calls.append({
                    "tool": api_type,
                    "arguments": {"query": query_terms[0]}
                })
        
        return tool_calls


class FixedAgent:
    def __init__(
        self,
        use_real_apis: bool = False,
        enable_tool_cache: bool = True,
        enable_workflow_cache: bool = True,
        enable_adaptive_ttl: bool = True,
        baseline_cache_system: Optional[Any] = None,
        use_redis: bool = True,
        enable_detailed_logging: bool = False
    ):
        self.use_real_apis = use_real_apis
        self.parser = ImprovedToolParser()
        self.enable_detailed_logging = enable_detailed_logging

        self._lock = threading.RLock()

        self.enable_tool_cache = enable_tool_cache
        self.enable_workflow_cache = enable_workflow_cache
        self.enable_adaptive_ttl = enable_adaptive_ttl

        self.metrics = CacheMetrics()
        self.llm_calls = 0
        self.questions_processed = 0
        self._category_lock = threading.RLock()
        self.category_metrics = defaultdict(lambda: {
            "tool_calls": 0,
            "cache_hits": 0,
            "execution_times": []
        })
        from cost_tracker import CostTracker
        self.cost_tracker = CostTracker()

        if baseline_cache_system:
            self.tool_cache = baseline_cache_system
            self.workflow_cache = None
            self.adaptive_manager = None
        else:
            if enable_adaptive_ttl:
                self.adaptive_manager = AdaptiveTTLManager()
            else:
                self.adaptive_manager = None
            
            if enable_tool_cache:
                self.tool_cache = DependencyAwareCache(
                    adaptive_ttl_manager=self.adaptive_manager if enable_adaptive_ttl else None,
                    use_redis=use_redis
                )
                if hasattr(self.tool_cache, 'cost_tracker'):
                    self.tool_cache.cost_tracker = self.cost_tracker

            else:
                self.tool_cache = None
            
            if enable_workflow_cache:
                self.workflow_cache = WorkflowCache(
                    default_ttl=300,
                    dependency_graph=self.tool_cache.dependency_graph if self.tool_cache else None
                )
                if self.tool_cache:
                    self.tool_cache.workflow_cache = self.workflow_cache
            else:
                self.workflow_cache = None
        
        self.execution_log = [] if enable_detailed_logging else None
    
    def _execute_tool(self, tool_name: str, arguments: Dict, session_id: str = None) -> Any:
        if tool_name in TOOLS:
            category = TOOLS[tool_name].category
            with self._category_lock:
                self.category_metrics[category]["tool_calls"] += 1


        cached_result = None
        if self.tool_cache:
            try:
                cached_result = self.tool_cache.get(tool_name, arguments, session_id)
            except Exception as e:
                logger.warning(f"Cache get error for {tool_name}: {e}")
                cached_result = None

        with self._lock:
            self.metrics.tool_calls_attempted += 1
            
            if cached_result is not None:
                self.metrics.tool_cache_hits += 1

                if tool_name in TOOLS:
                    category = TOOLS[tool_name].category
                    with self._category_lock:
                        self.category_metrics[category]["cache_hits"] += 1
                    
                    # Record cost only if not already tracked by baseline
                    if not (hasattr(self.tool_cache, 'cost_tracker') and 
                            self.tool_cache.cost_tracker is self.cost_tracker):
                        self.cost_tracker.record_api_call(tool_name, was_cached=True)
                
                logger.debug(f"CACHE HIT: {tool_name}")
                return cached_result
            
            self.metrics.tool_calls_executed += 1
        
        logger.debug(f"Executing: {tool_name}")
        
        start_time = time.time()
        try:
            if tool_name not in TOOLS:
                logger.error(f"Unknown tool: {tool_name}")
                return {"error": f"Unknown tool: {tool_name}"}
            
            tool = TOOLS[tool_name]
            result = tool.func(**arguments)
            execution_time = time.time() - start_time

            category = tool.category
            with self._category_lock:
                self.category_metrics[category]["tool_calls"] += 1  # Track ALL calls
                self.category_metrics[category]["execution_times"].append(execution_time)
                
                # Record cost only if not already tracked by baseline
                if not (hasattr(self.tool_cache, 'cost_tracker') and 
                        self.tool_cache.cost_tracker is self.cost_tracker):
                    self.cost_tracker.record_api_call(tool_name, was_cached=False)

        except Exception as e:
            logger.error(f"Tool execution error in {tool_name}: {e}")
            return {"error": str(e)}
        
        # Cache the result...
        if self.tool_cache and tool_name in TOOLS:
            try:
                tool_obj = TOOLS[tool_name]
                if not tool_obj.writes_to:
                    self.tool_cache.set(tool_name, arguments, result, session_id)
                else:
                    logger.info(f"Write operation detected: {tool_name}, invalidating dependencies")
                    self.tool_cache.invalidate_dependencies(tool_name, session_id)
            except Exception as e:
                logger.warning(f"Cache operation error for {tool_name}: {e}")
        
        return result
    
    def run_agent(self, question: str, session_id: str = None) -> Dict[str, Any]:

        with self._lock:
            self.llm_calls += 1
            self.questions_processed += 1
            self.metrics.workflow_queries += 1
        
        start_time = time.time()
        
        tool_calls = self.parser.parse_tool_calls(question)
        
        if not tool_calls:
            result = {
                "answer": "Processed without tools",
                "tool_calls": [],
                "execution_time": time.time() - start_time,
                "cache_hits": 0,
                "cache_misses": 0,
                "workflow_hit": False
            }
            
            if self.execution_log is not None:
                self.execution_log.append({
                    "question": question,
                    "tool_calls": [],
                    "answer": "No tools required",
                    "execution_time": result["execution_time"],
                    "cache_hits": 0,
                    "cache_misses": 0
                })
            
            return result
        
        unique_tool_calls = []
        seen = set()
        for tc in tool_calls:
            key = (tc["tool"], json.dumps(tc["arguments"], sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique_tool_calls.append(tc)
        
        workflow_hit = False
        cached_workflow_results = None
        
        if self.workflow_cache:
            try:
                workflow_key = tuple(
                    (tc["tool"], json.dumps(tc["arguments"], sort_keys=True)) 
                    for tc in unique_tool_calls
                )
                
                cached_workflow_results = self.workflow_cache.get_workflow(workflow_key, session_id)
                
                if cached_workflow_results is not None:
                    workflow_hit = True
                    with self._lock:
                        self.metrics.workflow_cache_hits += 1
                        self.metrics.tools_saved_by_workflow += len(unique_tool_calls)
                    
                    for tc in unique_tool_calls:
                        if tc["tool"] in TOOLS:
                            self.cost_tracker.record_api_call(tc["tool"], was_cached=True)

                    estimated_time_saved = sum(
                        TOOLS[tc["tool"]].cost / 1000.0
                        for tc in unique_tool_calls
                        if tc["tool"] in TOOLS
                    )
                    
                    with self._lock:
                        self.metrics.execution_time_saved += estimated_time_saved
                    
                    logger.info(f"WORKFLOW CACHE HIT: Saved {len(unique_tool_calls)} tool calls")
                    
                    result = {
                        "answer": "From workflow cache",
                        "tool_calls": unique_tool_calls,
                        "tool_results": cached_workflow_results,
                        "execution_time": time.time() - start_time,
                        "cache_hits": 0,
                        "cache_misses": 0,
                        "workflow_hit": True,
                        "tools_saved": len(unique_tool_calls)
                    }
                    
                    if self.execution_log is not None:
                        self.execution_log.append({
                            **result, 
                            "question": question, 
                            "timestamp": time.time()
                        })
                    
                    return result
            except Exception as e:
                logger.warning(f"Workflow cache error: {e}")
        
        tool_results = {}
        tools_before = self.metrics.tool_calls_attempted
        hits_before = self.metrics.tool_cache_hits
        
        for tool_call in unique_tool_calls:
            tool_name = tool_call["tool"]
            arguments = tool_call["arguments"]
            
            try:
                if tool_name == "calculate_distance" and "needs_locations" in arguments:
                    user_ids = arguments["needs_locations"]
                    locs = []
                    for uid in user_ids:
                        loc_key = f"get_user_location_{json.dumps({'user_id': uid}, sort_keys=True)}"
                        if loc_key in tool_results:
                            locs.append(tool_results[loc_key])
                        else:
                            loc = self._execute_tool("get_user_location", {"user_id": uid}, session_id)
                            locs.append(loc)
                            tool_results[loc_key] = loc
                    
                    if len(locs) >= 2 and "lat" in locs[0] and "lat" in locs[1]:
                        dist_args = {
                            "lat1": locs[0]["lat"],
                            "lon1": locs[0]["lon"],
                            "lat2": locs[1]["lat"],
                            "lon2": locs[1]["lon"]
                        }
                        result = self._execute_tool("calculate_distance", dist_args, session_id)
                        tool_results[f"{tool_name}_{user_ids[0]}_{user_ids[1]}"] = result
                else:
                    result = self._execute_tool(tool_name, arguments, session_id)
                    key = f"{tool_name}_{json.dumps(arguments, sort_keys=True)}"
                    tool_results[key] = result
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                tool_results[tool_name] = {"error": str(e)}
        
        tools_attempted_this_query = self.metrics.tool_calls_attempted - tools_before
        cache_hits_this_query = self.metrics.tool_cache_hits - hits_before
        cache_misses_this_query = tools_attempted_this_query - cache_hits_this_query
        
        if self.workflow_cache and not workflow_hit:
            try:
                workflow_key = tuple(
                    (tc["tool"], json.dumps(tc["arguments"], sort_keys=True)) 
                    for tc in unique_tool_calls
                )
                self.workflow_cache.cache_workflow(workflow_key, tool_results, session_id=session_id)
            except Exception as e:
                logger.warning(f"Workflow cache set error: {e}")
        
        execution_time = time.time() - start_time
        
        result = {
            "answer": "Processed with tools",
            "tool_calls": unique_tool_calls,
            "tool_results": tool_results,
            "execution_time": execution_time,
            "cache_hits": cache_hits_this_query,
            "cache_misses": cache_misses_this_query,
            "workflow_hit": workflow_hit
        }
        
        if self.execution_log is not None:
            self.execution_log.append({
                **result, 
                "question": question, 
                "timestamp": time.time()
            })
        
        return result
    
    def validate_stats(self):
        if self.metrics.tool_calls_attempted < self.metrics.tool_cache_hits:
            raise ValueError(
                f"BUG: cache hits ({self.metrics.tool_cache_hits}) > "
                f"attempted ({self.metrics.tool_calls_attempted})"
            )
        
        if self.metrics.tool_calls_attempted < self.metrics.tool_calls_executed:
            raise ValueError(
                f"BUG: executed ({self.metrics.tool_calls_executed}) > "
                f"attempted ({self.metrics.tool_calls_attempted})"
            )
        
        tool_hit_rate = self.metrics.get_tool_hit_rate()
        if tool_hit_rate > 100.01:
            raise ValueError(
                f"BUG: Tool hit rate {tool_hit_rate:.1f}% exceeds 100%. "
                f"Attempted: {self.metrics.tool_calls_attempted}, "
                f"Hits: {self.metrics.tool_cache_hits}"
            )
        
        logger.info(f"Validation passed: Tool hit rate {tool_hit_rate:.1f}%")
        return True
    
    def get_statistics(self) -> Dict:
        tool_hit_rate = self.metrics.get_tool_hit_rate()
        workflow_hit_rate = self.metrics.get_workflow_hit_rate()
        overall_efficiency = self.metrics.get_overall_efficiency()

        total_work_without_cache = (
            self.metrics.tool_calls_attempted + 
            self.metrics.tools_saved_by_workflow
        )
        
        total_work_avoided = (
            self.metrics.tool_cache_hits + 
            self.metrics.tools_saved_by_workflow
        )
        
        effective_hit_rate = (
            (total_work_avoided / total_work_without_cache * 100) 
            if total_work_without_cache > 0 else 0
        )
        
        hypothetical_tool_hit_rate = (
            (total_work_avoided / total_work_without_cache * 100)
            if total_work_without_cache > 0 else 0
        )
        
        stats = {
            "questions_processed": self.questions_processed,
            "llm_calls": self.llm_calls,
            
            "tool_calls_attempted": self.metrics.tool_calls_attempted,
            "tool_calls_executed": self.metrics.tool_calls_executed,
            "tool_cache_hits": self.metrics.tool_cache_hits,
            "tool_cache_hit_rate": f"{tool_hit_rate:.2f}%",
            
            "tool_cache_potential_hit_rate": f"{hypothetical_tool_hit_rate:.2f}%",
            "total_potential_tool_calls": total_work_without_cache,

            "workflow_queries": self.metrics.workflow_queries,
            "workflow_cache_hits": self.metrics.workflow_cache_hits,
            "workflow_cache_hit_rate": f"{workflow_hit_rate:.2f}%",
            "tools_saved_by_workflow": self.metrics.tools_saved_by_workflow,

            "effective_hit_rate": f"{effective_hit_rate:.2f}%",
            "total_work_avoided": total_work_avoided,
            "total_work_without_cache": total_work_without_cache,
            
            "overall_caching_efficiency": f"{overall_efficiency:.2f}%",
            "execution_time_saved_seconds": self.metrics.execution_time_saved,
            
            "category_breakdown": self._get_category_breakdown(),
            "cost_summary": self.cost_tracker.get_summary(),
            "cost_breakdown": self.cost_tracker.get_tool_breakdown()
        }
        
        if self.tool_cache and hasattr(self.tool_cache, 'get_stats'):
            try:
                stats["cache_backend_stats"] = self.tool_cache.get_stats()
            except:
                pass
        
        if self.adaptive_manager and hasattr(self.adaptive_manager, 'get_stats'):
            try:
                stats["adaptive_ttl"] = self.adaptive_manager.get_stats()
            except:
                pass
        
        return stats
    
    def _get_category_breakdown(self) -> Dict:
        with self._category_lock:
            category_snapshot = dict(self.category_metrics)
        
        breakdown = {}
        
        # If we have category data, use it
        if category_snapshot:
            for category, metrics in category_snapshot.items():
                total_calls = metrics["tool_calls"]
                cache_hits = metrics["cache_hits"]
                exec_times = metrics["execution_times"]
                
                breakdown[category] = {
                    "total_calls": total_calls,
                    "cache_hits": cache_hits,
                    "hit_rate": (cache_hits / total_calls * 100) if total_calls > 0 else 0.0,
                    "avg_execution_time": sum(exec_times) / len(exec_times) if exec_times else 0.0,
                }
        
        # Fallback: If no category data, generate from tools
        else:
            from tools import TOOLS
            
            # Aggregate from cost tracker if available
            if hasattr(self, 'cost_tracker') and self.cost_tracker:
                cost_breakdown = self.cost_tracker.get_tool_breakdown()
                
                category_totals = {}
                for tool_name, stats in cost_breakdown.items():
                    if tool_name in TOOLS:
                        category = TOOLS[tool_name].category
                        
                        if category not in category_totals:
                            category_totals[category] = {
                                "total_calls": 0,
                                "cache_hits": 0
                            }
                        
                        category_totals[category]["total_calls"] += stats["total_calls"]
                        category_totals[category]["cache_hits"] += stats["cached_calls"]
                
                for category, totals in category_totals.items():
                    total = totals["total_calls"]
                    hits = totals["cache_hits"]
                    breakdown[category] = {
                        "total_calls": total,
                        "cache_hits": hits,
                        "hit_rate": (hits / total * 100) if total > 0 else 0.0,
                        "avg_execution_time": 0.0,
                    }
        
        return breakdown
    
    def clear_execution_log(self):
        if self.execution_log is not None:
            self.execution_log.clear()
        logger.debug("Execution log cleared")
    
    def export_execution_log(self, filename: str = "execution_log.json"):
        if self.execution_log is None:
            logger.warning("Detailed logging is disabled, cannot export")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.execution_log, f, indent=2, default=str)
        logger.info(f"Exported execution log to {filename}")